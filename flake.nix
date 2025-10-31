{
  description = "Vulkan tutorial done in zig";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      perSystem =
        { system, pkgs, ... }:
        rec {
          formatter = pkgs.nixfmt-rfc-style;

          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              vulkan-headers
              vulkan-loader
              vulkan-validation-layers
              glslang
              zig
              lldb

              wayland
              wayland-protocols

              libxkbcommon
              xorg.libX11
            ];

            shellHook = ''
              mkdir -p ./vulkan
              cp -f ${pkgs.vulkan-headers}/share/vulkan/registry/vk.xml ./vulkan

              VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
              VK_LAYER_SETTINGS_PATH="$(pwd)./vk_layer_settings.txt"

              echo 'khronos_validation.enables = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED' > ./vk_layer_settings.txt
            '';
          };

          packages.waltuh = pkgs.stdenv.mkDerivation (finalAttrs: {
            pname = "waltuh";
            version = "69.420.1000001";

            src = pkgs.lib.cleanSource ./.;

            nativeBuildInputs = with pkgs; [
              zig
              zig.hook
              vulkan-headers
              glslang
              makeWrapper
            ];

            buildInputs = with pkgs; [
              vulkan-loader

              wayland
              wayland-protocols

              libxkbcommon
              xorg.libX11
            ];

            deps = pkgs.callPackage ./build.zig.zon.nix {
              name = "${finalAttrs.pname}-${finalAttrs.version}-deps";
            };
            strictDeps = true;

            zigBuildFlags = [
              "--system"
              "${finalAttrs.deps}"
            ];

            configurePhase = ''
              mkdir -p vulkan
              cp -f ${pkgs.vulkan-headers}/share/vulkan/registry/vk.xml ./vulkan

              substituteInPlace compile_shaders.sh \
                --replace-fail '/usr/bin/env sh' '${pkgs.bash}/bin/sh'
              ./compile_shaders.sh
            '';

            postInstall = ''
              wrapProgram $out/bin/waltuh \
                --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath finalAttrs.buildInputs}
            '';
          });

          packages.default = packages.waltuh;
        };
    };
}
