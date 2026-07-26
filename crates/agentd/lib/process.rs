//! Process lifecycle management for the agent daemon.
//!
//! Initializing [`ProcessManager`] transfers process-wide child-status ownership
//! to its dedicated thread: it drains `waitpid(-1, WNOHANG)` and therefore also
//! consumes statuses for untracked children. Code that needs an exit status must
//! acquire [`ProcessManager::spawn_guard`] before creating the process and finish
//! with [`ProcessSpawnGuard::track`]. It must not independently wait on the same
//! child PID during normal operation. Terminal teardown is the deliberate
//! exception: it may reap any remaining child directly when the manager itself
//! can no longer be assumed healthy.

use std::cell::Cell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, mpsc};
use std::task::{Context, Poll};
use std::thread;

use tokio::signal::unix::{Signal, SignalKind};
use tokio::sync::oneshot;

use crate::error::{AgentdError, AgentdResult};

//--------------------------------------------------------------------------------------------------
// Constants
//--------------------------------------------------------------------------------------------------

static PROCESS_MANAGER: OnceLock<Arc<ProcessManager>> = OnceLock::new();

std::thread_local! {
    static PROCESS_SPAWN_GUARD_HELD: Cell<bool> = const { Cell::new(false) };
}

//--------------------------------------------------------------------------------------------------
// Types
//--------------------------------------------------------------------------------------------------

/// Coordinates process spawning, exit observation, and process-wide child reaping.
pub struct ProcessManager {
    spawn_gate: RwLock<()>,
    state: Mutex<ProcessManagerState>,
    startup_error: OnceLock<String>,
    terminal_error: Mutex<Option<String>>,
}

struct ProcessManagerState {
    processes: HashMap<i32, oneshot::Sender<i32>>,
}

/// Keeps process reaping paused until a newly spawned PID is tracked.
pub struct ProcessSpawnGuard<'a> {
    manager: &'a ProcessManager,
    _spawn_gate: RwLockReadGuard<'a, ()>,
}

/// Observes the eventual exit code of a tracked process.
///
/// Processes terminated by a signal resolve to `-1`. If the reaper thread
/// unexpectedly drops the notification, the failure is logged and also resolves
/// to `-1` to preserve the exec-session wire protocol.
pub struct ProcessExitWatcher {
    pid: i32,
    receiver: oneshot::Receiver<i32>,
}

//--------------------------------------------------------------------------------------------------
// Methods
//--------------------------------------------------------------------------------------------------

impl ProcessManager {
    /// Returns the process-wide manager, starting its `SIGCHLD` thread on first use.
    ///
    /// The first call blocks synchronously until the dedicated thread has built
    /// its runtime and installed the `SIGCHLD` listener.
    ///
    /// # Errors
    ///
    /// Returns an error if the thread, runtime, or signal listener cannot start,
    /// or if the process manager has terminated unexpectedly.
    pub fn get() -> AgentdResult<Arc<Self>> {
        if let Some(manager) = PROCESS_MANAGER.get() {
            return manager.result();
        }

        let candidate = Arc::new(Self::new());
        let manager = PROCESS_MANAGER.get_or_init(move || {
            candidate.launch_thread();
            candidate
        });
        manager.result()
    }

    fn new() -> Self {
        Self {
            spawn_gate: RwLock::new(()),
            state: Mutex::new(ProcessManagerState::new()),
            startup_error: OnceLock::new(),
            terminal_error: Mutex::new(None),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test() -> Self {
        Self::new()
    }

    /// Opens a shared spawn section that must end by tracking the child PID.
    ///
    /// The returned guard permits other spawns concurrently but temporarily
    /// prevents the process manager from consuming an untracked fast exit.
    /// It must be acquired immediately before the OS spawn or fork operation.
    /// Acquiring it blocks synchronously while an exclusive reap pass owns the gate.
    ///
    /// # Errors
    ///
    /// Returns an error if the current thread already holds a spawn guard or the
    /// process manager has terminated.
    pub fn spawn_guard(&self) -> AgentdResult<ProcessSpawnGuard<'_>> {
        if PROCESS_SPAWN_GUARD_HELD.get() {
            return Err(AgentdError::ExecSession(
                "the current thread already holds a process spawn guard".to_string(),
            ));
        }

        let spawn_gate = self
            .spawn_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(error) = self
            .terminal_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return Err(AgentdError::ExecSession(error.clone()));
        }
        PROCESS_SPAWN_GUARD_HELD.set(true);
        Ok(ProcessSpawnGuard {
            manager: self,
            _spawn_gate: spawn_gate,
        })
    }

    fn launch_thread(self: &Arc<Self>) {
        let (startup_tx, startup_rx) = mpsc::sync_channel(1);
        let manager = Arc::clone(self);
        let spawn_result = thread::Builder::new()
            .name("agentd-process-manager".to_string())
            .spawn(move || {
                let failure_manager = Arc::clone(&manager);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    run_process_manager_thread(manager, startup_tx)
                }));
                let error = match result {
                    Ok(Err(error)) => error,
                    Ok(Ok(())) => "process manager thread stopped unexpectedly".to_string(),
                    Err(_) => "process manager thread panicked".to_string(),
                };
                failure_manager.fail(error);
            });

        let startup_result = match spawn_result {
            Ok(_) => startup_rx
                .recv()
                .unwrap_or_else(|error| Err(format!("receive thread startup: {error}"))),
            Err(error) => Err(format!("spawn process manager thread: {error}")),
        };
        if let Err(error) = startup_result {
            let _ = self.startup_error.set(error);
        }
    }

    fn result(self: &Arc<Self>) -> AgentdResult<Arc<Self>> {
        if let Some(error) = self.startup_error.get() {
            return Err(AgentdError::ExecSession(format!(
                "start process manager: {error}"
            )));
        }
        match self
            .terminal_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            Some(error) => Err(AgentdError::ExecSession(error.clone())),
            None => Ok(Arc::clone(self)),
        }
    }

    fn fail(&self, error: String) {
        let _reap_gate = self
            .spawn_gate
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        *self
            .terminal_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error);
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .processes
            .clear();
    }

    async fn run(self: Arc<Self>, mut signal: Signal) {
        self.reap_exited();
        while signal.recv().await.is_some() {
            self.reap_exited();
        }
    }

    fn reap_exited(&self) {
        let _reap_gate = self
            .spawn_gate
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.reap_exited();
    }
}

impl ProcessManagerState {
    fn new() -> Self {
        Self {
            processes: HashMap::new(),
        }
    }

    fn track(&mut self, pid: i32) -> AgentdResult<ProcessExitWatcher> {
        if pid <= 0 {
            return Err(AgentdError::ExecSession(format!(
                "cannot track invalid process PID {pid}"
            )));
        }

        let (sender, receiver) = oneshot::channel();
        match self.processes.entry(pid) {
            Entry::Vacant(entry) => {
                entry.insert(sender);
                Ok(ProcessExitWatcher { pid, receiver })
            }
            Entry::Occupied(_) => Err(AgentdError::ExecSession(format!(
                "process PID {pid} is already tracked"
            ))),
        }
    }

    fn reap_exited(&mut self) {
        loop {
            let mut status = 0;
            let pid = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
            if pid > 0 {
                if let Some(sender) = self.processes.remove(&pid) {
                    let _ = sender.send(exit_code(status));
                }
                continue;
            }
            if pid == 0 {
                break;
            }

            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            if error.raw_os_error() != Some(libc::ECHILD) {
                eprintln!("agentd: waitpid failed while reaping processes: {error}");
            }
            break;
        }
    }
}

impl ProcessSpawnGuard<'_> {
    /// Tracks the spawned PID before allowing process reaping to proceed.
    ///
    /// Consuming the guard makes the manager the sole owner of that PID's exit
    /// status during normal operation. Await the returned [`ProcessExitWatcher`]
    /// instead of calling `waitpid` or a child handle's wait method. Terminal
    /// teardown may bypass the manager as a last-resort fallback.
    ///
    /// # Errors
    ///
    /// Returns an error when `pid` is not positive or is already tracked.
    pub fn track(self, pid: i32) -> AgentdResult<ProcessExitWatcher> {
        self.manager
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .track(pid)
    }
}

//--------------------------------------------------------------------------------------------------
// Trait Implementations
//--------------------------------------------------------------------------------------------------

impl Drop for ProcessSpawnGuard<'_> {
    fn drop(&mut self) {
        PROCESS_SPAWN_GUARD_HELD.set(false);
    }
}

impl Future for ProcessExitWatcher {
    type Output = i32;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(cx) {
            Poll::Ready(Ok(code)) => Poll::Ready(code),
            Poll::Ready(Err(error)) => {
                eprintln!(
                    "agentd: process manager dropped the exit notification for PID {}: {error}",
                    self.pid
                );
                Poll::Ready(-1)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Functions
//--------------------------------------------------------------------------------------------------

fn run_process_manager_thread(
    manager: Arc<ProcessManager>,
    startup: mpsc::SyncSender<Result<(), String>>,
) -> Result<(), String> {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let error = format!("build process manager Tokio runtime: {error}");
            let _ = startup.send(Err(error.clone()));
            return Err(error);
        }
    };

    runtime.block_on(async move {
        match tokio::signal::unix::signal(SignalKind::child()) {
            Ok(signal) => {
                let _ = startup.send(Ok(()));
                manager.run(signal).await;
                Err("process manager SIGCHLD listener closed".to_string())
            }
            Err(error) => {
                let error = format!("install process manager SIGCHLD listener: {error}");
                let _ = startup.send(Err(error.clone()));
                Err(error)
            }
        }
    })
}

fn exit_code(status: i32) -> i32 {
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        -1
    }
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::sync::{Arc, Barrier, TryLockError, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;

    const HELPER_ENV: &str = "MSB_AGENTD_PROCESS_MANAGER_HELPER";
    const HELPER_SENTINEL: &str = "process-manager-helper-passed";
    const TEST_NAME: &str =
        "process::tests::concurrent_spawns_are_tracked_before_exclusive_reaping";

    #[test]
    fn concurrent_spawns_are_tracked_before_exclusive_reaping() {
        if std::env::var_os(HELPER_ENV).is_some() {
            run_concurrent_reap_scenario();
            println!("{HELPER_SENTINEL}");
            return;
        }

        let mut helper = Command::new(std::env::current_exe().expect("current test binary"))
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(HELPER_ENV, "1")
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn isolated process manager test");
        let mut output = String::new();
        helper
            .stdout
            .take()
            .expect("helper stdout")
            .read_to_string(&mut output)
            .expect("read helper stdout");

        match helper.wait() {
            Ok(status) => assert!(status.success(), "helper failed: {status}\n{output}"),
            Err(error) if error.raw_os_error() == Some(libc::ECHILD) => {}
            Err(error) => panic!("wait for helper: {error}"),
        }
        assert!(
            output.contains(HELPER_SENTINEL),
            "helper did not complete the reap scenario:\n{output}"
        );
    }

    #[test]
    fn invalid_pids_are_rejected() {
        let manager = ProcessManager::new();
        for pid in [-1, 0] {
            let error = match manager
                .spawn_guard()
                .expect("acquire process spawn guard")
                .track(pid)
            {
                Ok(_) => panic!("invalid PID should be rejected"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(&pid.to_string()));
        }
    }

    #[test]
    fn nested_spawn_guards_are_rejected() {
        let manager = ProcessManager::new();
        let first = manager
            .spawn_guard()
            .expect("acquire first process spawn guard");
        let error = match manager.spawn_guard() {
            Ok(_) => panic!("nested process spawn guard should be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("already holds"));

        drop(first);
        drop(
            manager
                .spawn_guard()
                .expect("guard should be available after drop"),
        );
    }

    #[test]
    fn terminal_failure_rejects_spawns_and_wakes_exits() {
        let manager = Arc::new(ProcessManager::new());
        let exit_watcher = manager
            .spawn_guard()
            .expect("acquire process spawn guard")
            .track(12345)
            .expect("track test PID");

        manager.fail("process manager test failure".to_string());

        assert!(manager.result().is_err());
        assert!(manager.spawn_guard().is_err());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime");
        assert_eq!(runtime.block_on(exit_watcher), -1);
    }

    fn run_concurrent_reap_scenario() {
        let manager = Arc::new(ProcessManager::new());
        let spawned = Arc::new(Barrier::new(3));
        let release_tracking = Arc::new(Barrier::new(3));
        let (exit_watcher_tx, exit_watcher_rx) = mpsc::channel();

        let first = spawn_tracked_process(
            Arc::clone(&manager),
            Arc::clone(&spawned),
            Arc::clone(&release_tracking),
            exit_watcher_tx.clone(),
            41,
        );
        let second = spawn_tracked_process(
            Arc::clone(&manager),
            Arc::clone(&spawned),
            Arc::clone(&release_tracking),
            exit_watcher_tx,
            42,
        );

        spawned.wait();
        assert!(matches!(
            manager.spawn_gate.try_write(),
            Err(TryLockError::WouldBlock)
        ));

        let (reaped_tx, reaped_rx) = mpsc::channel();
        let manager_thread = {
            let manager = Arc::clone(&manager);
            thread::spawn(move || {
                manager.reap_exited();
                reaped_tx.send(()).expect("report reap completion");
            })
        };

        assert!(
            reaped_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "exclusive reaping entered while spawn guards were active"
        );

        release_tracking.wait();
        let first_exit_watcher = exit_watcher_rx.recv().expect("first exit watcher");
        let second_exit_watcher = exit_watcher_rx.recv().expect("second exit watcher");
        reaped_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("process manager should proceed after both processes are tracked");

        first.join().expect("first spawn thread");
        second.join().expect("second spawn thread");
        manager_thread.join().expect("process manager thread");

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime");
        let (first_code, second_code) =
            runtime.block_on(async { tokio::join!(first_exit_watcher, second_exit_watcher) });
        let mut codes = [first_code, second_code];
        codes.sort_unstable();
        assert_eq!(codes, [41, 42]);

        let orphan = Command::new("/bin/sh")
            .args(["-c", "exit 43"])
            .spawn()
            .expect("spawn untracked child");
        let orphan_pid = orphan.id() as i32;
        drop(orphan);
        wait_until_exited_without_reaping(orphan_pid);

        manager.reap_exited();
        assert_already_reaped(orphan_pid);
    }

    fn spawn_tracked_process(
        manager: Arc<ProcessManager>,
        spawned: Arc<Barrier>,
        release_tracking: Arc<Barrier>,
        exit_watcher_tx: mpsc::Sender<ProcessExitWatcher>,
        code: i32,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let guard = manager.spawn_guard().expect("acquire process spawn guard");
            let child = Command::new("/bin/sh")
                .args(["-c", &format!("exit {code}")])
                .spawn()
                .expect("spawn tracked child");
            let pid = child.id() as i32;
            drop(child);

            spawned.wait();
            release_tracking.wait();

            let exit_watcher = guard.track(pid).expect("track child");
            exit_watcher_tx
                .send(exit_watcher)
                .expect("send tracked exit watcher");
        })
    }

    fn wait_until_exited_without_reaping(pid: i32) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let mut info = unsafe { std::mem::zeroed::<libc::siginfo_t>() };
            let ret = unsafe {
                libc::waitid(
                    libc::P_PID,
                    pid as libc::id_t,
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            assert_eq!(ret, 0, "waitid failed: {}", std::io::Error::last_os_error());
            if unsafe { info.si_pid() } == pid {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }

        panic!("child {pid} did not exit");
    }

    fn assert_already_reaped(pid: i32) {
        let ret = unsafe { libc::waitpid(pid, std::ptr::null_mut(), libc::WNOHANG) };
        assert_eq!(ret, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
    }
}
