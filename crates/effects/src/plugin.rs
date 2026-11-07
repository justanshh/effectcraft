//! The effect plug-in API, versions [`MIN_PLUGIN_API_VERSION`] through [`PLUGIN_API_VERSION`].
//!
//! A plug-in is anything implementing [`EffectPlugin`]: a [`PluginManifest`] (stable id,
//! display name, Effects & Presets category, typed parameters) and a `render` that processes a
//! [`PluginFrame`] in place. [`register_plugin`] turns it into an ordinary [`EffectSpec`]:
//! instances are property groups like every built-in effect, so plug-in parameters animate,
//! take expressions, show in Effect Controls and are addressable by agents, and the effect is
//! applied by id (`effect.apply {"effect": "org.example.posterize"}`) like a built-in.
//!
//! Plug-ins come from Rust code (implement the trait and register it at start-up) or from
//! sandboxed WebAssembly modules (`effectcraft-plugin`, see `docs/plugins.md`), which implement
//! the same contract through a small C-like ABI.
//!
//! The contract (stable within an API version):
//! * pixels are premultiplied RGBA `f32` (0–1, may exceed 1 in 32 bpc projects), row-major,
//!   in the layer's buffer, which may be scaled (preview resolution) and padded;
//! * parameters are flattened to `f64`s in declaration order: slider / angle / checkbox (0 or 1)
//!   / popup (0-based index) take one value, point two (**frame pixels**, already mapped to the
//!   buffer), colour four (RGBA 0–1);
//! * `time` is layer time in seconds; `scale` is the buffer's pixels per layer pixel (scale
//!   pixel distances like blur radii by it);
//! * rendering must be deterministic (same inputs, same output) and may run on several threads
//!   at once.
//!
//! API 2 adds [`EffectPlugin::render_with_neighbours`]: the frames immediately around `time` on
//! the layer's frame grid (`prev` / `next`), as [`PluginNeighbours`]. They come from the host's
//! [`EffectHost::self_at`](crate::EffectHost::self_at), so they carry the same masks and the
//! effects *below* the plug-in and remain a pure function of the project. A module opts in with
//! `"api": 2` and exports `ec_render2`; API 1 modules keep loading unchanged and never see
//! neighbours.

use std::sync::{Arc, OnceLock, RwLock};

use effectcraft_keyframe::Value;
use effectcraft_project::ParamUi;
use serde::{Deserialize, Serialize};

use crate::{Buf, EffectCtx, EffectSpec, Image, ParamSpec, RenderFn};

/// The newest plug-in API version this build implements.
pub const PLUGIN_API_VERSION: u32 = 2;

/// The oldest plug-in API version this build still loads.
pub const MIN_PLUGIN_API_VERSION: u32 = 1;

/// Whether this build implements plug-in API version `v`.
pub const fn supports_api(v: u32) -> bool {
    v >= MIN_PLUGIN_API_VERSION && v <= PLUGIN_API_VERSION
}

/// The most plug-in effects one process can register.
pub const MAX_PLUGINS: usize = 256;

/// A plug-in parameter's kind, default and range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PluginParamKind {
    Slider {
        default: f64,
        #[serde(default)]
        min: f64,
        #[serde(default = "hundred")]
        max: f64,
        /// Slider range (defaults to min..max).
        #[serde(default, rename = "sliderMin")]
        slider_min: Option<f64>,
        #[serde(default, rename = "sliderMax")]
        slider_max: Option<f64>,
        #[serde(default = "two")]
        decimals: u8,
    },
    Angle {
        #[serde(default)]
        default: f64,
    },
    Checkbox {
        #[serde(default)]
        default: bool,
    },
    Popup {
        options: Vec<String>,
        /// 0-based index.
        #[serde(default)]
        default: u32,
    },
    /// Default as fractions of the layer size (`[0.5, 0.5]` = centre).
    Point {
        #[serde(default = "centre")]
        default: [f64; 2],
    },
    /// RGBA, 0–1.
    Color {
        #[serde(default = "white")]
        default: [f64; 4],
    },
}

fn hundred() -> f64 {
    100.0
}
fn two() -> u8 {
    2
}
fn centre() -> [f64; 2] {
    [0.5, 0.5]
}
fn white() -> [f64; 4] {
    [1.0; 4]
}

impl PluginParamKind {
    /// How many `f64`s the parameter flattens to.
    pub fn width(&self) -> usize {
        match self {
            PluginParamKind::Point { .. } => 2,
            PluginParamKind::Color { .. } => 4,
            _ => 1,
        }
    }
}

/// One parameter of a plug-in effect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PluginParam {
    /// Stable id (the property's match name; letters, digits, `_`).
    pub id: String,
    pub name: String,
    #[serde(flatten)]
    pub kind: PluginParamKind,
}

/// What a plug-in declares about itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PluginManifest {
    /// The plug-in API version it was written for (supported: [`MIN_PLUGIN_API_VERSION`] through
    /// [`PLUGIN_API_VERSION`]).
    pub api: u32,
    /// Stable, reverse-domain effect id (`org.example.posterize`); `ec.` is reserved.
    pub id: String,
    pub name: String,
    /// Effects & Presets category: one of [`crate::CATEGORIES`] or a new one.
    pub category: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub params: Vec<PluginParam>,
}

impl PluginManifest {
    /// Check the manifest against this API version.
    pub fn validate(&self) -> Result<(), String> {
        const MAX_PARAMS: usize = 256;
        if !supports_api(self.api) {
            return Err(format!(
                "plug-in `{}` targets plug-in API {} but this build implements API {MIN_PLUGIN_API_VERSION}–{PLUGIN_API_VERSION}",
                self.id, self.api
            ));
        }
        let ok_id = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
        if !ok_id(&self.id) || self.id.starts_with("ec.") {
            return Err(format!("bad plug-in id `{}` (reverse-domain, letters, digits, `.`, `_`, `-`; `ec.` is reserved)", self.id));
        }
        if self.name.trim().is_empty() || self.category.trim().is_empty() {
            return Err(format!("plug-in `{}` needs a name and a category", self.id));
        }
        if self.params.len() > MAX_PARAMS {
            return Err(format!("plug-in `{}` has {} parameters (at most {MAX_PARAMS})", self.id, self.params.len()));
        }
        let mut seen = std::collections::HashSet::new();
        for p in &self.params {
            if !Self::valid_param_id(&p.id) || !seen.insert(p.id.as_str()) {
                return Err(format!("plug-in `{}`: bad or duplicate parameter id `{}`", self.id, p.id));
            }
            if let PluginParamKind::Popup { options, default } = &p.kind
                && (options.is_empty() || *default as usize >= options.len())
            {
                return Err(format!("plug-in `{}`: popup `{}` needs options and a default index in range", self.id, p.id));
            }
            if let PluginParamKind::Slider { default, min, max, slider_min, slider_max, .. } = &p.kind {
                let (lo, hi) = (slider_min.unwrap_or(*min), slider_max.unwrap_or(*max));
                if [*default, *min, *max, lo, hi].iter().any(|v| !v.is_finite()) || min > max || lo > hi || default < min || default > max {
                    return Err(format!("plug-in `{}`: slider `{}` needs min ≤ default ≤ max (and sliderMin ≤ sliderMax)", self.id, p.id));
                }
            }
        }
        Ok(())
    }

    /// Total flattened parameter count.
    pub fn flat_len(&self) -> usize {
        self.params.iter().map(|p| p.kind.width()).sum()
    }

    /// A parameter id: `letters/digits/_`, or `/`-separated group path segments of those. The
    /// host nests `borders/autoScale` into twirl-down groups (see
    /// [`instantiate`](crate::instantiate)), so plug-ins can declare grouped UI with paths.
    fn valid_param_id(id: &str) -> bool {
        !id.is_empty() && !id.starts_with('/') && !id.ends_with('/') && id.split('/').all(|s| s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
    }
}

/// The pixels a plug-in renders, in place.
pub struct PluginFrame<'a> {
    pub width: u32,
    pub height: u32,
    /// Premultiplied RGBA, row-major, `width * height` pixels.
    pub pixels: &'a mut [[f32; 4]],
    /// Buffer pixels per layer pixel (preview resolution).
    pub scale: f64,
}

/// One neighbouring frame of a plug-in's own layer (plug-in API 2).
#[derive(Clone, Copy)]
pub struct PluginNeighbour<'a> {
    /// Layer time of this frame, in seconds.
    pub time: f64,
    /// Premultiplied RGBA, row-major, on the same grid as the frame being rendered
    /// (`width * height` pixels: same size, scale and padding).
    pub pixels: &'a [[f32; 4]],
}

/// The frames immediately around the one a plug-in renders (plug-in API 2).
///
/// The host passes the layer's frames on its frame grid: at a frame boundary, the frame before
/// and after it; between frames, the two frames that bracket `time`. A neighbour is `None` when
/// the layer has no pixels there (the start or end of its source) or the host cannot provide
/// one; a plug-in must degrade gracefully. API 1 modules implement
/// [`EffectPlugin::render`] and never receive these.
#[derive(Clone, Copy, Default)]
pub struct PluginNeighbours<'a> {
    pub prev: Option<PluginNeighbour<'a>>,
    pub next: Option<PluginNeighbour<'a>>,
}

/// Parameter values for one render, flattened (see the module docs) with typed accessors.
pub struct PluginParams<'a> {
    pub manifest: &'a PluginManifest,
    pub flat: &'a [f64],
}

impl PluginParams<'_> {
    fn slot(&self, id: &str) -> Option<&[f64]> {
        let mut at = 0;
        for p in &self.manifest.params {
            let w = p.kind.width();
            if p.id == id {
                return self.flat.get(at..at + w);
            }
            at += w;
        }
        None
    }
    /// Slider / angle / popup index / checkbox (0 or 1).
    pub fn f(&self, id: &str) -> f64 {
        self.slot(id).and_then(|s| s.first().copied()).unwrap_or(0.0)
    }
    pub fn b(&self, id: &str) -> bool {
        self.f(id) != 0.0
    }
    /// Point in frame pixels.
    pub fn point(&self, id: &str) -> [f64; 2] {
        self.slot(id).map(|s| [s[0], s[1]]).unwrap_or([0.0; 2])
    }
    pub fn color(&self, id: &str) -> [f64; 4] {
        self.slot(id).map(|s| [s[0], s[1], s[2], s[3]]).unwrap_or([1.0; 4])
    }
}

/// An effect plug-in.
pub trait EffectPlugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
    /// Process `frame` in place. `time` is layer time in seconds. An error leaves the frame as
    /// it was (the error is logged once).
    fn render(&self, frame: &mut PluginFrame, params: &PluginParams, time: f64) -> Result<(), String>;
    /// API 2: process `frame` in place with the frames around `time` available. The default
    /// ignores them and calls [`EffectPlugin::render`], so API 1 plug-ins need no change.
    fn render_with_neighbours(&self, frame: &mut PluginFrame, params: &PluginParams, time: f64, neighbours: &PluginNeighbours<'_>) -> Result<(), String> {
        let _ = neighbours;
        self.render(frame, params, time)
    }
    /// Where the plug-in came from (a file path, `built-in`…), for listings.
    fn source(&self) -> String {
        "built-in".into()
    }
}

static SLOTS: [OnceLock<Arc<dyn EffectPlugin>>; MAX_PLUGINS] = [const { OnceLock::new() }; MAX_PLUGINS];

/// Registered plug-in effects, in registration order.
fn plugin_specs() -> &'static RwLock<Vec<&'static EffectSpec>> {
    static P: OnceLock<RwLock<Vec<&'static EffectSpec>>> = OnceLock::new();
    P.get_or_init(|| RwLock::new(vec![]))
}

/// The registered plug-in effects.
pub fn plugins() -> Vec<&'static EffectSpec> {
    plugin_specs().read().map(|v| v.clone()).unwrap_or_default()
}

/// The plug-in behind a registered effect id.
pub fn plugin(id: &str) -> Option<Arc<dyn EffectPlugin>> {
    SLOTS.iter().filter_map(|s| s.get()).find(|p| p.manifest().id == id).cloned()
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn param_spec(p: &PluginParam) -> ParamSpec {
    let (default, ui) = match &p.kind {
        PluginParamKind::Slider { default, min, max, slider_min, slider_max, decimals } => (
            Value::Scalar(*default),
            ParamUi::Slider { min: *min, max: *max, slider_min: slider_min.unwrap_or(*min), slider_max: slider_max.unwrap_or(*max), decimals: *decimals },
        ),
        PluginParamKind::Angle { default } => (Value::Scalar(*default), ParamUi::Angle),
        PluginParamKind::Checkbox { default } => (Value::Bool(*default), ParamUi::Checkbox),
        PluginParamKind::Popup { options, default } => (Value::Enum(*default), ParamUi::Popup { options: options.clone() }),
        PluginParamKind::Point { default } => (Value::Vec2(*default), ParamUi::Point),
        PluginParamKind::Color { default } => (Value::Color(*default), ParamUi::Color),
    };
    ParamSpec { id: leak(&p.id), name: leak(&p.name), default, ui }
}

/// Register a plug-in effect: it appears in Effects & Presets under its category and is
/// applied by id like a built-in. Fails for a bad manifest, a taken id or when
/// [`MAX_PLUGINS`] are registered.
pub fn register_plugin(plugin: Arc<dyn EffectPlugin>) -> Result<&'static EffectSpec, String> {
    let m = plugin.manifest().clone();
    m.validate()?;
    let mut specs = plugin_specs().write().map_err(|_| "plug-in registry poisoned".to_string())?;
    if crate::registry().iter().any(|s| s.id == m.id) || specs.iter().any(|s| s.id == m.id) {
        return Err(format!("an effect with id `{}` is already registered", m.id));
    }
    let slot = SLOTS
        .iter()
        .position(|s| s.get().is_none())
        .ok_or_else(|| format!("too many plug-in effects: a session registers at most {MAX_PLUGINS}; remove some plug-ins and restart EffectCraft"))?;
    SLOTS[slot].set(plugin).map_err(|_| "plug-in slot taken".to_string())?;
    let spec: &'static EffectSpec = Box::leak(Box::new(EffectSpec {
        id: leak(&m.id),
        name: leak(&m.name),
        category: leak(&m.category),
        params: m.params.iter().map(param_spec).collect(),
        render: TRAMPOLINES[slot],
        gpu: false,
        float: true,
    }));
    specs.push(spec);
    Ok(spec)
}

/// Each slot needs its own `fn` (effect render functions are plain function pointers).
fn trampoline<const I: usize>(ctx: &EffectCtx, buf: Buf) -> Buf {
    run_slot(I, ctx, buf)
}

macro_rules! trampolines {
    ($($i:literal)*) => { [$(trampoline::<$i> as RenderFn),*] };
}

const TRAMPOLINES: [RenderFn; MAX_PLUGINS] = trampolines!(
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
    64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
    96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
    128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
    160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
    192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
    224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255
);

/// Flatten an instance's evaluated parameters for the plug-in (points mapped to frame pixels).
pub fn flatten(m: &PluginManifest, ctx: &EffectCtx, buf: &Buf) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.flat_len());
    for p in &m.params {
        let v = ctx.params.get(&p.id);
        match &p.kind {
            PluginParamKind::Point { .. } => {
                let q = v.map(Value::as_vec2).unwrap_or([0.0; 2]);
                let (x, y) = buf.to_px(q);
                out.extend([x, y]);
            }
            PluginParamKind::Color { .. } => {
                let c = v.map(Value::as_color).unwrap_or([1.0; 4]);
                out.extend(c.iter().map(|x| *x as f64));
            }
            PluginParamKind::Checkbox { .. } => out.push(if v.map(Value::as_bool).unwrap_or(false) { 1.0 } else { 0.0 }),
            PluginParamKind::Popup { .. } => out.push(v.map(Value::as_enum).unwrap_or(0) as f64),
            _ => out.push(v.map(Value::as_f64).unwrap_or(0.0)),
        }
    }
    out
}

/// The layer times of the frames around `time` on a `fps` grid: the frame before and after it
/// at a frame boundary, the two frames that bracket it between frames.
fn neighbour_times(time: f64, fps: f64) -> (f64, f64) {
    let step = 1.0 / fps;
    let at = time * fps;
    let before = at.floor();
    let after = at.ceil();
    if after - before < 1e-9 { (time - step, time + step) } else { (before / fps, after / fps) }
}

/// A neighbour resampled onto the current buffer's grid (same pixels, scale and padding).
fn same_grid(nb: Buf, b: &Buf) -> Image {
    if nb.img.width == b.img.width && nb.img.height == b.img.height && nb.offset == b.offset && nb.scale == b.scale {
        return nb.img;
    }
    let k = nb.scale / b.scale.max(1e-9);
    crate::util::gen_image(b.img.width, b.img.height, |x, y| {
        let lx = (x as f64 + 0.5 - b.offset[0]) * k + nb.offset[0];
        let ly = (y as f64 + 0.5 - b.offset[1]) * k + nb.offset[1];
        nb.img.sample_bilinear(lx, ly)
    })
}

/// The neighbours a plug-in may read (`None` without a host, or at the layer's ends).
fn neighbour_images(ctx: &EffectCtx, buf: &Buf) -> (Option<(f64, Image)>, Option<(f64, Image)>) {
    let Some(host) = ctx.env.host else { return (None, None) };
    let (prev_time, next_time) = neighbour_times(ctx.time, ctx.fps());
    let at = |t: f64| host.self_at(t, ctx.env.effect_index).map(|nb| (t, same_grid(nb, buf)));
    (at(prev_time), at(next_time))
}

fn run_slot(slot: usize, ctx: &EffectCtx, mut buf: Buf) -> Buf {
    let Some(plugin) = SLOTS[slot].get() else { return buf };
    let m = plugin.manifest();
    let flat = flatten(m, ctx, &buf);
    let params = PluginParams { manifest: m, flat: &flat };
    let (width, height) = (buf.img.width, buf.img.height);
    let before = buf.img.data.clone();
    // API 2 plug-ins may read the frames around this one. Fetch them before borrowing the
    // pixels mutably, and only for modules that asked for them.
    let (prev, next) = if m.api >= 2 { neighbour_images(ctx, &buf) } else { (None, None) };
    let neighbours = PluginNeighbours {
        prev: prev.as_ref().map(|(time, img)| PluginNeighbour { time: *time, pixels: &img.data }),
        next: next.as_ref().map(|(time, img)| PluginNeighbour { time: *time, pixels: &img.data }),
    };
    let mut frame = PluginFrame { width, height, pixels: &mut buf.img.data, scale: buf.scale };
    if let Err(e) = plugin.render_with_neighbours(&mut frame, &params, ctx.time, &neighbours) {
        log::warn!("plug-in `{}`: {e}", m.id);
        buf.img.data = before;
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use effectcraft_project::build::Ids;

    use crate::{EffectEnv, EffectHost, Image, LayerPixels, flatten_params, instantiate};

    struct Recorder {
        m: PluginManifest,
        seen: Mutex<Vec<[f64; 4]>>,
    }

    impl EffectPlugin for Recorder {
        fn manifest(&self) -> &PluginManifest {
            &self.m
        }

        fn render(&self, _frame: &mut PluginFrame, _params: &PluginParams, _time: f64) -> Result<(), String> {
            Ok(())
        }

        fn render_with_neighbours(&self, frame: &mut PluginFrame, _params: &PluginParams, _time: f64, neighbours: &PluginNeighbours<'_>) -> Result<(), String> {
            let red = |n: Option<PluginNeighbour<'_>>| n.map_or(f64::NAN, |n| f64::from(n.pixels[0][0]));
            if let Ok(mut seen) = self.seen.lock() {
                seen.push([
                    neighbours.prev.map_or(f64::NAN, |n| n.time),
                    neighbours.next.map_or(f64::NAN, |n| n.time),
                    red(neighbours.prev),
                    red(neighbours.next),
                ]);
            }
            frame.pixels[0] = [1.0, 2.0, 3.0, 4.0];
            Ok(())
        }
    }

    struct Neighbours;

    impl EffectHost for Neighbours {
        fn layer(&self, _id: u64, _masks_and_effects: bool) -> Option<LayerPixels> {
            None
        }

        fn audio(&self, _id: u64, _start: f64, _frames: usize, _rate: u32) -> Option<Vec<f32>> {
            None
        }

        fn self_at(&self, layer_time: f64, _effects: usize) -> Option<Buf> {
            // The red channel carries the time, so a test can prove which frames arrived.
            Some(Buf { img: Image::filled(2, 2, [layer_time as f32, 0.0, 0.0, 1.0]), offset: [0.0; 2], scale: 1.0 })
        }
    }

    fn recorder(id: &str) -> (Arc<Recorder>, &'static EffectSpec) {
        let m = PluginManifest {
            api: 2,
            id: id.into(),
            name: "Recorder".into(),
            category: "Test Plug-ins".into(),
            version: "1".into(),
            author: String::new(),
            description: String::new(),
            params: vec![],
        };
        let plugin = Arc::new(Recorder { m, seen: Mutex::new(vec![]) });
        let spec = register_plugin(plugin.clone()).expect("a free plug-in slot");
        (plugin, spec)
    }

    fn render_with_host(spec: &EffectSpec, time: f64, env: EffectEnv<'_>) -> Image {
        let mut next = 1;
        let g = instantiate(spec, &mut Ids(&mut next), spec.name, [2.0, 2.0]);
        let params = flatten_params(&g, &mut |p| p.value.clone());
        let ctx = EffectCtx { params: &params, time, layer_size: [2.0, 2.0], seed: 1, adjustment: false, env };
        (spec.render)(&ctx, Buf { img: Image::filled(2, 2, [0.5; 4]), offset: [0.0; 2], scale: 1.0 }).img
    }

    #[test]
    fn neighbour_times_bracket_the_layer_time() {
        // On a frame: the frame before and after it.
        assert_eq!(neighbour_times(0.5, 4.0), (0.25, 0.75));
        // Between frames: the two frames it lies between.
        let (prev, next) = neighbour_times(0.6, 4.0);
        assert!((prev - 0.5).abs() < 1e-12 && (next - 0.75).abs() < 1e-12, "{prev} {next}");
        // Floating-point land just below a frame boundary still brackets it.
        let (prev, next) = neighbour_times(0.5 - 1e-12, 4.0);
        assert!((prev - 0.25).abs() < 1e-12 && (next - 0.5).abs() < 1e-12, "{prev} {next}");
    }

    #[test]
    fn api2_plugins_get_the_hosts_neighbour_frames() {
        let (plugin, spec) = recorder("org.test.recorder.host");
        let img = render_with_host(spec, 0.5, EffectEnv { host: Some(&Neighbours), frame_rate: 4.0, effect_index: 0, ..Default::default() });
        assert_eq!(img.get(0, 0), [1.0, 2.0, 3.0, 4.0], "the plug-in rendered");
        let seen = plugin.seen.lock().expect("not poisoned");
        assert_eq!(seen.len(), 1);
        // At 0.5 s and 4 fps the neighbours are 0.25 s and 0.75 s, and their pixels are theirs.
        assert_eq!(seen[0], [0.25, 0.75, 0.25, 0.75]);
    }

    #[test]
    fn api2_plugins_without_a_host_get_no_neighbours() {
        let (plugin, spec) = recorder("org.test.recorder.nohost");
        let img = render_with_host(spec, 0.5, EffectEnv { frame_rate: 4.0, ..Default::default() });
        assert_eq!(img.get(0, 0), [1.0, 2.0, 3.0, 4.0], "the plug-in still rendered");
        let seen = plugin.seen.lock().expect("not poisoned");
        assert!(seen[0][0].is_nan() && seen[0][1].is_nan(), "{:?}", seen[0]);
    }
}
