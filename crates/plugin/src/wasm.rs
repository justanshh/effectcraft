//! The wasmi-backed [`EffectPlugin`] (plug-in API 1 and 2, see the crate docs).

use std::sync::Mutex;

use wasmi::{Config, Engine, Instance, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc, WasmParams, WasmResults};

use crate::{EffectPlugin, MIN_PLUGIN_API_VERSION, PLUGIN_API_VERSION, PluginFrame, PluginManifest, PluginNeighbour, PluginNeighbours, PluginParams};

/// Instruction budget per frame: a fixed allowance plus some per pixel. Generous for per-pixel
/// work, but a runaway loop stops instead of hanging the render.
const FUEL_BASE: u64 = 50_000_000;
const FUEL_PER_PIXEL: u64 = 20_000;
/// The most linear memory a plug-in instance may have (a 4K float frame is ~130 MB).
const MAX_MEMORY: usize = 1 << 30;
/// The longest manifest read.
const MAX_MANIFEST: i32 = 1 << 20;

/// API 1: one buffer.
type RenderFn = TypedFunc<(i32, i32, i32, i32, i32, f64, f64), i32>;
/// API 2: the same call plus the previous and next frame buffers and their layer times.
type Render2Fn = TypedFunc<(i32, i32, i32, i32, i32, f64, f64, i32, f64, i32, f64, i32), i32>;

enum Renderer {
    V1(RenderFn),
    V2(Render2Fn),
}

/// One instantiated module (renders running on several threads each take their own).
struct Inst {
    store: Store<StoreLimits>,
    instance: Instance,
    memory: Memory,
    alloc: TypedFunc<i32, i32>,
    render: Renderer,
}

/// An instantiated module before its API version (and so its render export) is known.
struct BaseInst {
    store: Store<StoreLimits>,
    instance: Instance,
    memory: Memory,
    alloc: TypedFunc<i32, i32>,
}

impl BaseInst {
    fn with_api(self, api: u32) -> Result<Inst, String> {
        let render = match api {
            1 => Renderer::V1(self.instance.get_typed_func::<_, i32>(&self.store, "ec_render").map_err(|_| missing("ec_render"))?),
            2 => Renderer::V2(self.instance.get_typed_func::<_, i32>(&self.store, "ec_render2").map_err(|_| missing("ec_render2"))?),
            v => return Err(unsupported(v)),
        };
        Ok(Inst { store: self.store, instance: self.instance, memory: self.memory, alloc: self.alloc, render })
    }
}

impl Inst {
    fn func<P: WasmParams, R: WasmResults>(&self, name: &str) -> Result<TypedFunc<P, R>, String> {
        self.instance.get_typed_func::<P, R>(&self.store, name).map_err(|_| missing(name))
    }
}

fn missing(name: &str) -> String {
    format!("the module exports no `{name}` (or with the wrong signature)")
}

fn unsupported(v: u32) -> String {
    format!("the module implements plug-in API {v}, this build implements API {MIN_PLUGIN_API_VERSION}–{PLUGIN_API_VERSION}")
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn instantiate(engine: &Engine, module: &Module) -> Result<BaseInst, String> {
    let mut store = Store::new(engine, StoreLimitsBuilder::new().memory_size(MAX_MEMORY).instances(1).build());
    store.limiter(|l| l);
    store.set_fuel(FUEL_BASE).map_err(err)?;
    let linker = Linker::<StoreLimits>::new(engine);
    let instance = linker.instantiate_and_start(&mut store, module).map_err(|e| format!("instantiation failed: {e}"))?;
    let memory = instance.get_memory(&store, "memory").ok_or("the module exports no `memory`")?;
    let alloc = instance.get_typed_func::<i32, i32>(&store, "ec_alloc").map_err(|_| missing("ec_alloc"))?;
    Ok(BaseInst { store, instance, memory, alloc })
}

/// A loaded WebAssembly effect plug-in.
pub struct WasmPlugin {
    manifest: PluginManifest,
    source: String,
    engine: Engine,
    module: Module,
    /// The ABI version the module implements (1 or 2), checked against its manifest at load.
    api: u32,
    /// Idle instances.
    pool: Mutex<Vec<Inst>>,
}

impl WasmPlugin {
    pub fn new(bytes: &[u8], source: &str) -> Result<WasmPlugin, String> {
        let mut config = Config::default();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, bytes).map_err(|e| format!("not a valid WebAssembly module: {e}"))?;
        if let Some(i) = module.imports().next() {
            return Err(format!("plug-ins can't import anything (the module imports `{}.{}`)", i.module(), i.name()));
        }
        let mut inst = instantiate(&engine, &module)?;
        let version = inst.instance.get_typed_func::<(), i32>(&inst.store, "ec_api_version").map_err(|_| missing("ec_api_version"))?;
        let v = version.call(&mut inst.store, ()).map_err(err)?;
        if !(MIN_PLUGIN_API_VERSION as i32..=PLUGIN_API_VERSION as i32).contains(&v) {
            return Err(format!("the module implements plug-in API {v}, this build implements API {MIN_PLUGIN_API_VERSION}–{PLUGIN_API_VERSION}"));
        }
        let api = v as u32;
        let inst = inst.with_api(api)?;
        let inst = inst_with_manifest(inst, api)?;
        Ok(WasmPlugin { manifest: inst.0, source: source.to_string(), engine, module, api, pool: Mutex::new(vec![inst.1]) })
    }

    fn take(&self) -> Result<Inst, String> {
        if let Some(i) = self.pool.lock().ok().and_then(|mut p| p.pop()) {
            return Ok(i);
        }
        instantiate(&self.engine, &self.module)?.with_api(self.api)
    }

    fn give_back(&self, inst: Inst) {
        if let Ok(mut p) = self.pool.lock() {
            p.push(inst);
        }
    }

    fn render_v1(&self, frame: &mut PluginFrame, params: &PluginParams, time: f64) -> Result<(), String> {
        let n = frame.pixels.len();
        let params_bytes = params.flat.len() * 8;
        let total = params_bytes + n * 16;
        if total > i32::MAX as usize {
            return Err("frame too large for a 32-bit plug-in".into());
        }
        let mut inst = self.take()?;
        inst.store.set_fuel(FUEL_BASE + FUEL_PER_PIXEL * n as u64).map_err(err)?;
        let base = inst.alloc.call(&mut inst.store, total as i32).map_err(err)? as u32 as usize;
        let mut buf = Vec::with_capacity(total);
        for v in params.flat {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        push_pixels(&mut buf, frame.pixels);
        inst.memory.write(&mut inst.store, base, &buf).map_err(|e| format!("ec_alloc returned a buffer out of bounds: {e}"))?;
        let px_ptr = (base + params_bytes) as i32;
        let r = match &inst.render {
            Renderer::V1(f) => f
                .call(&mut inst.store, (px_ptr, frame.width as i32, frame.height as i32, base as i32, params.flat.len() as i32, time, frame.scale))
                .map_err(|e| format!("ec_render trapped: {e}")),
            Renderer::V2(_) => return Err("the module implements API 2 but the manifest declares API 1".into()),
        };
        let r = match r {
            Ok(0) => read_pixels(&mut inst, base + params_bytes, &mut buf, frame),
            Ok(code) => Err(format!("ec_render returned error {code}")),
            Err(e) => Err(e),
        };
        if r.is_ok() {
            self.give_back(inst);
        }
        r
    }

    fn render_v2(&self, frame: &mut PluginFrame, params: &PluginParams, time: f64, neighbours: &PluginNeighbours<'_>) -> Result<(), String> {
        let n = frame.pixels.len();
        let params_bytes = params.flat.len() * 8;
        let frame_bytes = n * 16;
        // Parameters, then the current frame, then the previous and next frames. The neighbour
        // buffers are always allocated (zeroed when there is none), so a module can read them
        // without a null check; `flags` says which hold real frames.
        let total = params_bytes + frame_bytes * 3;
        if total > i32::MAX as usize {
            return Err("frame too large for a 32-bit plug-in".into());
        }
        let prev = neighbours.prev.as_ref().filter(|p| p.pixels.len() == n);
        let next = neighbours.next.as_ref().filter(|p| p.pixels.len() == n);
        let mut inst = self.take()?;
        inst.store.set_fuel(FUEL_BASE + FUEL_PER_PIXEL * (n * 3) as u64).map_err(err)?;
        let base = inst.alloc.call(&mut inst.store, total as i32).map_err(err)? as u32 as usize;
        let mut buf = Vec::with_capacity(total);
        for v in params.flat {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        push_pixels(&mut buf, frame.pixels);
        push_neighbour(&mut buf, prev, frame_bytes);
        push_neighbour(&mut buf, next, frame_bytes);
        inst.memory.write(&mut inst.store, base, &buf).map_err(|e| format!("ec_alloc returned a buffer out of bounds: {e}"))?;
        let px_ptr = (base + params_bytes) as i32;
        let prev_ptr = (base + params_bytes + frame_bytes) as i32;
        let next_ptr = (base + params_bytes + frame_bytes * 2) as i32;
        let flags = i32::from(prev.is_some()) | (i32::from(next.is_some()) << 1);
        let prev_time = prev.map_or(0.0, |p| p.time);
        let next_time = next.map_or(0.0, |p| p.time);
        let r = match &inst.render {
            Renderer::V2(f) => f
                .call(
                    &mut inst.store,
                    (
                        px_ptr,
                        frame.width as i32,
                        frame.height as i32,
                        base as i32,
                        params.flat.len() as i32,
                        time,
                        frame.scale,
                        prev_ptr,
                        prev_time,
                        next_ptr,
                        next_time,
                        flags,
                    ),
                )
                .map_err(|e| format!("ec_render2 trapped: {e}")),
            Renderer::V1(_) => return Err("the module implements API 1 but the manifest declares API 2".into()),
        };
        let r = match r {
            Ok(0) => read_pixels(&mut inst, base + params_bytes, &mut buf, frame),
            Ok(code) => Err(format!("ec_render2 returned error {code}")),
            Err(e) => Err(e),
        };
        if r.is_ok() {
            self.give_back(inst);
        }
        r
    }
}

/// Read the manifest, validate it, and check it declares the ABI version the module implements.
fn inst_with_manifest(mut inst: Inst, api: u32) -> Result<(PluginManifest, Inst), String> {
    let ptr = inst.func::<(), i32>("ec_manifest_ptr")?.call(&mut inst.store, ()).map_err(err)?;
    let len = inst.func::<(), i32>("ec_manifest_len")?.call(&mut inst.store, ()).map_err(err)?;
    if !(0..=MAX_MANIFEST).contains(&len) {
        return Err(format!("bad manifest length {len}"));
    }
    let mut text = vec![0u8; len as usize];
    inst.memory.read(&inst.store, ptr as u32 as usize, &mut text).map_err(|e| format!("manifest out of bounds: {e}"))?;
    let manifest: PluginManifest = serde_json::from_slice(&text).map_err(|e| format!("bad manifest JSON: {e}"))?;
    manifest.validate()?;
    if manifest.api != api {
        return Err(format!("the module implements plug-in API {api} but its manifest declares API {} (they must match)", manifest.api));
    }
    Ok((manifest, inst))
}

fn push_pixels(buf: &mut Vec<u8>, pixels: &[[f32; 4]]) {
    for px in pixels {
        for c in px {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
}

/// Append a neighbour frame, or `bytes` zeros when this side has none.
fn push_neighbour(buf: &mut Vec<u8>, neighbour: Option<&PluginNeighbour<'_>>, bytes: usize) {
    match neighbour {
        Some(p) => push_pixels(buf, p.pixels),
        None => buf.resize(buf.len() + bytes, 0),
    }
}

/// Read the pixels a successful render left at `at` back into `frame`.
fn read_pixels(inst: &mut Inst, at: usize, buf: &mut Vec<u8>, frame: &mut PluginFrame) -> Result<(), String> {
    let n = frame.pixels.len();
    buf.resize(at + n * 16, 0);
    inst.memory.read(&inst.store, at, &mut buf[at..]).map_err(err)?;
    for (px, b) in frame.pixels.iter_mut().zip(buf[at..].as_chunks::<16>().0.iter()) {
        for (c, v) in px.iter_mut().zip(b.as_chunks::<4>().0.iter()) {
            *c = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
        }
    }
    Ok(())
}

impl EffectPlugin for WasmPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn source(&self) -> String {
        self.source.clone()
    }

    fn render(&self, frame: &mut PluginFrame, params: &PluginParams, time: f64) -> Result<(), String> {
        self.render_with_neighbours(frame, params, time, &PluginNeighbours::default())
    }

    fn render_with_neighbours(&self, frame: &mut PluginFrame, params: &PluginParams, time: f64, neighbours: &PluginNeighbours<'_>) -> Result<(), String> {
        match self.manifest.api {
            1 => self.render_v1(frame, params, time),
            _ => self.render_v2(frame, params, time, neighbours),
        }
    }
}
