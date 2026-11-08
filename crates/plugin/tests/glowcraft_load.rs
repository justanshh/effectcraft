//! Loads the built GlowCraft plug-in (DeepGlow-EC) through the real host path: manifest
//! deserialisation, validation and a default-parameter render. Run after `build.ps1` produces
//! `DeepGlow-EC/target/wasm32-unknown-unknown/release/deep_glow.wasm`.
//!
//! Tolerates both shipped variants: the flat v2.0 manifest, and the grouped v2.1 one (id-path
//! groups, see DeepGlow-EC/manifests/grouped-ui.json) once the host supports `/` ids.

#[test]
fn glowcraft_wasm_loads_and_previews() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../DeepGlow/DeepGlow-EC/target/wasm32-unknown-unknown/release/deep_glow.wasm");
    // The artifact lives outside this repository; skip (rather than fail) when it isn't built.
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("skipping: {path} is not built");
        return;
    };

    // preview() runs the full load path: serde manifest parse, validate(), instant render.
    let (manifest, frames) =
        effectcraft_plugin::preview(&bytes, 48, 32, &vec![[0.1, 0.2, 0.3, 1.0]; 48 * 32], &[0.0]).expect("GlowCraft.wasm should load and render");

    assert_eq!(manifest["name"], "GlowCraft");
    let params = manifest["params"].as_array().expect("params");
    let ids: Vec<&str> = params.iter().filter_map(|p| p["id"].as_str()).collect();
    let grouped = ["uiMain/glowMode", "uiStyle/uiGamma/sceneGamma"];
    let flat = ["glowMode", "sceneGamma"];
    let matched = grouped.iter().all(|g| ids.iter().any(|i| i == g)) || flat.iter().all(|f| ids.iter().any(|i| i == f));
    assert!(matched, "manifest ids match neither flat nor grouped layout");
    assert_eq!(frames.len(), 1);
    assert!(frames[0].iter().all(|p| p.iter().all(|v| v.is_finite())));
}
