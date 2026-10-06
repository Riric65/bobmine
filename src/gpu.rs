//! Front CUDA du mineur : API Driver (libcuda.so), pas de cudart à installer.
//!
//! Le PTX est compilé au build time par build.rs et embarqué dans le binaire.
//! Le contexte et le module sont créés une seule fois (le JIT du PTX coûte cher)
//! puis réutilisés à chaque challenge.
//!
//! Les fonctions unsafe de l'API Driver sont factorisées dans `sys` et
//! documentées; le reste du fichier est safe Rust.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::os::raw::c_int;
use std::ptr;
use std::sync::Mutex;

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;

unsafe extern "C" {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(buf: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: CUdevice) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, module: CUmodule, name: *const c_char)
        -> CUresult;
    fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        stream: *mut c_void,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuGetErrorString(result: CUresult, p_str: *mut *const c_char) -> CUresult;
}

/// PTX du kernel, compilé par build.rs (absent si build sans nvcc).
#[cfg(cuda)]
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/miner.ptx"));

/// Constantes du kernel + point d'entrée, à jour pour un challenge donné.
pub struct Gpu {
    ctx: CUcontext,
    f_mine: CUfunction,
    d_pre: CUdeviceptr,
    d_found: CUdeviceptr,
    grid: u32,
    block: u32,
    /// Nombre d'indices traités par lancement (réactivité + mesure).
    pub batch: u64,
    pub device_name: String,
}

// Le contexte CUDA est manipulé depuis un seul worker à la fois (mine()
// lance et joint son thread GPU avant le challenge suivant).
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

fn err_str(code: CUresult) -> String {
    unsafe {
        let mut p: *const c_char = ptr::null();
        if cuGetErrorString(code, &mut p) == CUDA_SUCCESS && !p.is_null() {
            return CStr::from_ptr(p).to_string_lossy().into_owned();
        }
    }
    format!("code {code}")
}

/// Garde les erreurs explicites plutôt que des codes bruts.
fn check(code: CUresult, what: &str) -> Result<(), String> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {}", err_str(code)))
    }
}

impl Gpu {
    /// Initialise CUDA et charge le kernel une seule fois par processus.
    pub fn global() -> &'static Result<Mutex<Gpu>, String> {
        static GPU: std::sync::OnceLock<Result<Mutex<Gpu>, String>> =
            std::sync::OnceLock::new();
        GPU.get_or_init(Self::init)
    }

    fn init() -> Result<Mutex<Gpu>, String> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;

            let mut name_buf = [0u8; 128];
            check(
                cuDeviceGetName(name_buf.as_mut_ptr() as *mut c_char, 128, dev),
                "cuDeviceGetName",
            )?;
            let device_name = CStr::from_ptr(name_buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned();

            let mut sm = 0;
            check(
                cuDeviceGetAttribute(&mut sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev),
                "cuDeviceGetAttribute",
            )?;

            let mut ctx: CUcontext = ptr::null_mut();
            check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;

            let image = CString::new(PTX).map_err(|e| format!("PTX: {e}"))?;
            let mut module: CUmodule = ptr::null_mut();
            check(
                cuModuleLoadData(&mut module, image.as_ptr() as *const c_void),
                "cuModuleLoadData (JIT du PTX)",
            )?;

            let f_name = CString::new("bob_mine").unwrap();
            let mut f_mine: CUfunction = ptr::null_mut();
            check(
                cuModuleGetFunction(&mut f_mine, module, f_name.as_ptr()),
                "cuModuleGetFunction(bob_mine)",
            )?;

            let mut d_pre: CUdeviceptr = 0;
            let mut sz = 0usize;
            let pre_name = CString::new("c_pre").unwrap();
            check(
                cuModuleGetGlobal_v2(&mut d_pre, &mut sz, module, pre_name.as_ptr()),
                "cuModuleGetGlobal(c_pre)",
            )?;

            let mut d_k: CUdeviceptr = 0;
            let k_name = CString::new("c_K").unwrap();
            check(
                cuModuleGetGlobal_v2(&mut d_k, &mut sz, module, k_name.as_ptr()),
                "cuModuleGetGlobal(c_K)",
            )?;
            check(cuMemcpyHtoD_v2(d_k, K32.as_ptr() as *const c_void, K32.len() * 4), "cuMemcpyHtoD(c_K)")?;

            let mut d_found: CUdeviceptr = 0;
            check(
                cuMemAlloc_v2(&mut d_found, 8),
                "cuMemAlloc(found)",
            )?;

            let block = 256u32;
            let grid = (sm.max(1) as u32) * 8;
            let threads = grid as u64 * block as u64;

            Ok(Mutex::new(Gpu {
                ctx,
                f_mine,
                d_pre,
                d_found,
                grid,
                block,
                batch: threads * 4096,
                device_name,
            }))
        }
    }

    /// Prépare le GPU pour un challenge : contexte courant, prefix, found vide.
    pub fn prepare(&mut self, prefix: &str) -> Result<(), String> {
        if prefix.len() != 48 {
            return Err(format!(
                "le kernel CUDA est spécialisé pour un prefix de 48 octets (reçu: {})",
                prefix.len()
            ));
        }
        let mut pre = [0u32; 12];
        let b = prefix.as_bytes();
        for (i, w) in pre.iter_mut().enumerate() {
            *w = u32::from_be_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
        }
        unsafe {
            check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")?;
            check(
                cuMemcpyHtoD_v2(
                    self.d_pre,
                    pre.as_ptr() as *const c_void,
                    pre.len() * 4,
                ),
                "cuMemcpyHtoD(c_pre)",
            )?;
            let sentinel = u64::MAX;
            check(
                cuMemcpyHtoD_v2(
                    self.d_found,
                    &sentinel as *const u64 as *const c_void,
                    8,
                ),
                "cuMemcpyHtoD(found)",
            )?;
        }
        Ok(())
    }

    /// Lance le kernel sur `[start, start+count)`. Retourne l'indice gagnant
    /// éventuel (`None` si le lot est terminé sans solution).
    pub fn run_batch(&mut self, start: u64, count: u64, bits: u32) -> Result<Option<u64>, String> {
        let mut p_start = start;
        let mut p_count = count;
        let mut p_found = self.d_found;
        let mut p_bits = bits;
        let mut params = [
            &mut p_start as *mut u64 as *mut c_void,
            &mut p_count as *mut u64 as *mut c_void,
            &mut p_found as *mut u64 as *mut c_void,
            &mut p_bits as *mut u32 as *mut c_void,
        ];
        unsafe {
            check(
                cuLaunchKernel(
                    self.f_mine,
                    self.grid,
                    1,
                    1,
                    self.block,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            check(cuCtxSynchronize(), "cuCtxSynchronize")?;
            let mut found = u64::MAX;
            check(
                cuMemcpyDtoH_v2(
                    &mut found as *mut u64 as *mut c_void,
                    self.d_found,
                    8,
                ),
                "cuMemcpyDtoH(found)",
            )?;
            if found == u64::MAX {
                Ok(None)
            } else {
                Ok(Some(found))
            }
        }
    }
}

/// Constantes SHA-256, copiées dans la mémoire constante du module.
const K32: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
