#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

# The wheel installs its FFmpeg source under the system interpreter's data path.
data_path=$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_path("data"))')
package_path=$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/PyNvVideoCodec
source_archive=${2:-${data_path}/external/ffmpeg/src/ffmpeg-9.0.1.tar.xz}
package_path=${1:-${package_path}}
patch_path=${3:-/opt/ffmpeg-rtsp-redirect-limit.patch}

test -f "${source_archive}"
test -f "${package_path}/libavformat.so"
test -f "${package_path}/libavutil.so"
test -f "${patch_path}"
printf '%s  %s\n' \
    cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635 \
    "${source_archive}" | sha256sum --check --status

build_dir=$(mktemp -d)
trap 'rm -rf "${build_dir}"' EXIT
tar -xJf "${source_archive}" -C "${build_dir}"
source_dir=${build_dir}/ffmpeg-9.0.1
patch --batch --forward -p1 -d "${source_dir}" < "${patch_path}"

cd "${source_dir}"
./configure \
    --prefix="${build_dir}/install" \
    --disable-gpl --disable-nonfree --enable-static --disable-shared \
    --enable-pic --disable-autodetect --disable-x86asm --disable-everything \
    --disable-programs --disable-doc --disable-encoders --disable-decoders \
    --disable-hwaccels --disable-swresample --disable-swscale \
    --disable-avfilter --disable-avdevice --disable-libdrm \
    --enable-demuxers --enable-muxers --enable-protocols \
    --enable-parsers --enable-bsfs --enable-avformat --enable-avcodec \
    --enable-avutil --arch="$(uname -m)" --extra-cflags=-fvisibility=default
make -j16 libavformat/libavformat.a libavcodec/libavcodec.a libavutil/libavutil.a

# Match the wheel's layout: avcodec is embedded in libavformat, while avutil
# remains a separate shared library. Keep libavformat.so.63 for both architectures.
gcc -shared -Wl,-soname,libavformat.so.63 -Wl,-z,defs '-Wl,-rpath,$ORIGIN' \
    -o "${build_dir}/libavformat.so" \
    -Wl,--whole-archive libavformat/libavformat.a libavcodec/libavcodec.a \
    -Wl,--no-whole-archive -L"${package_path}" -lavutil -lz -lm -lpthread

nm -D --defined-only "${package_path}/libavformat.so" | awk '{print $3}' | sort -u > "${build_dir}/old-symbols"
nm -D --defined-only "${build_dir}/libavformat.so" | awk '{print $3}' | sort -u > "${build_dir}/new-symbols"
if ! cmp -s "${build_dir}/old-symbols" "${build_dir}/new-symbols"; then
    echo "Patched libavformat exports differ from the PyNvVideoCodec wheel" >&2
    exit 1
fi

for name in libavformat.so libavformat.so.63 libavformat.so.63.1.101; do
    test -f "${package_path}/${name}"
    install -m 0755 "${build_dir}/libavformat.so" "${package_path}/${name}"
done
