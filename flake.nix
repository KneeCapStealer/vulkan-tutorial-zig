{
  description = "Vulkan tutorial done in zig";

  nixConfig = {
    extra-substituters = [
      "https://chaotic-nyx.cachix.org/"
    ];
    extra-trusted-public-keys = [
      "chaotic-nyx.cachix.org-1:HfnXSw4pj95iI/n17rIDy40agHj12WfF+Gqk6SonIT8="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    chaotic.url = "github:chaotic-cx/nyx/nyxpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      perSystem =
        { system, pkgs, ... }:
        {
          # Use chaotic overlay
          # _module.args.pkgs = import inputs.nixpkgs {
          #   inherit system;
          #   overlays = [ inputs.chaotic.overlays.default ];
          # };

          formatter = pkgs.nixfmt-rfc-style;

          devShells.default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                vulkan-headers
                vulkan-loader
                vulkan-validation-layers
                glslang
              ]
              ++ (with pkgs; [
                zig
                glfw
              ]);

            shellHook = ''
              mkdir -p ./vulkan
              cp -f ${pkgs.vulkan-headers}/share/vulkan/registry/vk.xml ./vulkan
            '';
          };
        };
    };
}
