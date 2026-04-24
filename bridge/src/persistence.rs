use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::sync::Mutex;
use lazy_static::lazy_static;

use crate::trajectories::KernelXEvent;

/* C FFI Bindings */
extern "C" {
    fn ht_create(size: i32) -> *mut c_void;
    fn ht_set(ht: *mut c_void, key: *const c_void, klen: usize, value: *const c_void, vlen: usize, expires_at: i64);
    fn ht_free(ht: *mut c_void);
    fn aof_open(filename: *const c_char) -> i32;
    fn aof_append_set(key: *const c_void, klen: usize, value: *const c_void, vlen: usize, expires_at: i64);
    fn radish_export_json(ht: *mut c_void, filename: *const c_char) -> i32;
    fn aof_get_size() -> usize;
}

pub fn get_aof_size() -> u64 {
    unsafe { aof_get_size() as u64 }
}

pub enum RadishMode {
    Perception = 0,
    Training = 1,
}

struct RadishDB {
    ht: *mut c_void,
}

unsafe impl Send for RadishDB {}

lazy_static! {
    static ref DB: Mutex<Option<RadishDB>> = Mutex::new(None);
}

pub fn init(aof_path: &str) -> Result<(), String> {
    let mut db_lock = DB.lock().unwrap();
    if db_lock.is_some() {
        return Ok(());
    }

    unsafe {
        let ht = ht_create(1024); // Large initial size for KernelX
        if ht.is_null() {
            return Err("Failed to create RadishDB Hashtable".to_string());
        }

        let c_path = CString::new(aof_path).unwrap();
        if aof_open(c_path.as_ptr()) == 0 {
            ht_free(ht);
            return Err(format!("Failed to open RadishDB AOF at {}", aof_path));
        }

        *db_lock = Some(RadishDB { ht });
    }
    
    println!("[RadishDB] WAL initialized at {}", aof_path);
    Ok(())
}

pub fn persist_event(event: &KernelXEvent) {
    let db_lock = DB.lock().unwrap();
    if let Some(ref db) = *db_lock {
        unsafe {
            // Key: PID (4 bytes)
            let key = &event.pid as *const u32 as *const c_void;
            // Value: The full KernelXEvent struct (208 bytes)
            let val = event as *const KernelXEvent as *const c_void;
            
            // Log to In-Memory Hash Table
            ht_set(db.ht, key, 4, val, std::mem::size_of::<KernelXEvent>(), 0);
            
            // Log to Write-Ahead Log (AOF)
            aof_append_set(key, 4, val, std::mem::size_of::<KernelXEvent>(), 0);
        }
    }
}

pub fn export_to_json(path: &str) -> Result<(), String> {
    let db_lock = DB.lock().unwrap();
    if let Some(ref db) = *db_lock {
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
