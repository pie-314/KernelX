use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::sync::Mutex;
use lazy_static::lazy_static;

use crate::trajectories::KernelXEvent;

/* C FFI Bindings - from radish_ffi.h */
extern "C" {
    fn radish_ht_create(initial_size: i32) -> *mut c_void;
    fn radish_ht_set(ht: *mut c_void, key: *const c_void, klen: usize, value: *const c_void, vlen: usize, expires_at: i64);
    fn radish_ht_free(ht: *mut c_void);
    fn radish_aof_open(filename: *const c_char) -> i32;
    fn radish_aof_append(key: *const c_void, klen: usize, value: *const c_void, vlen: usize, expires_at: i64);
    fn radish_export_json(ht: *mut c_void, filename: *const c_char) -> i32;
    fn radish_aof_get_size() -> usize;
}

pub fn get_aof_size() -> u64 {
    unsafe { radish_aof_get_size() as u64 }
}

struct RadishDB {
    ht: *mut c_void,
}

unsafe impl Send for RadishDB {}

struct PersistenceState {
    db: Option<RadishDB>,
}

unsafe impl Send for PersistenceState {}

lazy_static! {
    static ref STATE: Mutex<PersistenceState> = Mutex::new(PersistenceState {
        db: None,
    });
}

pub fn init(aof_path: &str) -> Result<(), String> {
    let mut state = STATE.lock().unwrap();
    if state.db.is_some() {
        return Ok(());
    }

    unsafe {
        let ht = radish_ht_create(1024);
        if ht.is_null() {
            return Err("Failed to create RadishDB Hashtable".to_string());
        }

        let c_path = CString::new(aof_path).unwrap();
        if radish_aof_open(c_path.as_ptr()) == 0 {
            radish_ht_free(ht);
            return Err(format!("Failed to open RadishDB AOF at {}", aof_path));
        }

        state.db = Some(RadishDB { ht });
    }
    
    println!("[RadishDB] WAL initialized at {}", aof_path);
    Ok(())
}

pub fn persist_event(event: &KernelXEvent) {
    let state = STATE.lock().unwrap();
    
    if let Some(ref db) = state.db {
        unsafe {
            // Key: PID (4 bytes)
            let key = &event.pid as *const u32 as *const c_void;
            // Value: The full KernelXEvent struct (208 bytes)
            let val = event as *const KernelXEvent as *const c_void;
            
            // Log to In-Memory Hash Table
            radish_ht_set(db.ht, key, 4, val, std::mem::size_of::<KernelXEvent>(), 0);
            
            // Log to Write-Ahead Log (AOF)
            radish_aof_append(key, 4, val, std::mem::size_of::<KernelXEvent>(), 0);
        }
    }
}

pub fn export_to_json(path: &str) -> Result<(), String> {
    let state = STATE.lock().unwrap();
    if let Some(ref db) = state.db {
        let c_path = CString::new(path).unwrap();
        unsafe {
            if radish_export_json(db.ht, c_path.as_ptr()) == 1 {
                Ok(())
            } else {
                Err(format!("Failed to export JSON to {}", path))
            }
        }
    } else {
        Err("Database not initialized".to_string())
    }
}
