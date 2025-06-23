#!/usr/bin/env sh

mkdir -p shaders/out

glslang -V100 shaders/triangle.vert -o shaders/out/vert.spv
glslang -V100 shaders/triangle.frag -o shaders/out/frag.spv
