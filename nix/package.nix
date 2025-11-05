{
  lib,
  callPackage,
  stdenvNoCC,
  bash,
  zig,
  vulkan-headers,
  vulkan-loader,
  glslang,
  makeWrapper,
  wayland,
  wayland-protocols,
  libxkbcommon,
  libX11
}:
stdenvNoCC.mkDerivation (finalAttrs: {
  pname = "waltuh";
  version = "69.420.1000001";

  src = lib.cleanSource ../.;

  nativeBuildInputs = [
    zig
    zig.hook
    vulkan-headers
    glslang
    makeWrapper
  ];

  buildInputs = [
    vulkan-loader

    wayland
    wayland-protocols

    libxkbcommon
    libX11
  ];

  deps = callPackage ./build.zig.zon.nix {
    name = "${finalAttrs.pname}-${finalAttrs.version}-deps";
  };
  strictDeps = true;

  zigBuildFlags = [
    "--system"
    "${finalAttrs.deps}"
  ];

  configurePhase = ''
    mkdir -p vulkan
    cp -f ${vulkan-headers}/share/vulkan/registry/vk.xml ./vulkan

    substituteInPlace compile_shaders.sh \
      --replace-fail '/usr/bin/env sh' '${bash}/bin/sh'
    ./compile_shaders.sh
  '';

  postInstall = ''
    wrapProgram $out/bin/waltuh \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath finalAttrs.buildInputs}
  '';
})

