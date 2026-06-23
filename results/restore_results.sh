#!/bin/bash

files=(
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/1.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/2.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/3.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/4.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/10.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/11.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/12.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/13.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/30.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/31.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/40.json.tar.bz2"
    "cache-rotated_surface_code-X-ufrust-grow20dB-v2-cachedivs/41.json.tar.bz2"
)

for f in "${files[@]}" ; do
    cat ${f}.* > ${f}
    tar -jxvf ${f}
done
