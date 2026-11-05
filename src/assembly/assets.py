"""Verified upstream assets and a portable build of its unmodified contact math."""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'data/vendor/assembly_nist'
PLUGIN = ROOT / 'data/runtime/assembly/libassemblenist_sdf.so'


def bootstrap():
    if not (VENDOR / 'UPSTREAM.json').is_file():
        raise FileNotFoundError('Precision assembly assets missing. Run: python -m experiments.assembly_setup')
    if not PLUGIN.is_file():
        raise FileNotFoundError('Contact plugin missing. Run: python -m experiments.assembly_setup --build-only')
    os.environ['ASSEMBLENIST_SDF_PLUGIN'] = str(PLUGIN)
    os.environ.setdefault('ASSEMBLENIST_SDF_DIR', str(VENDOR / 'assets/sdf'))
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def verify():
    manifest = json.loads((VENDOR / 'UPSTREAM.json').read_text())
    for entry in manifest['files']:
        path = VENDOR / entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f"Modified upstream asset: {entry['path']}")
    return manifest


def build_plugin():
    import mujoco
    source = (VENDOR / 'plugin/assemblenist_sdf.cc').read_text()
    source = source.replace('#include <immintrin.h>', '')
    original = 'inline float h2f(uint16_t x) { return _cvtsh_ss(x); }'
    portable = '''inline float h2f(uint16_t x) {
      uint32_t sign = (uint32_t(x & 0x8000)) << 16;
      uint32_t exp = (x >> 10) & 31, mant = x & 1023, bits;
      if (exp == 0) return (x & 0x8000 ? -1.f : 1.f) * std::ldexp(float(mant), -24);
      if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
      else bits = sign | ((exp + 112) << 23) | (mant << 13);
      float result; std::memcpy(&result, &bits, 4); return result;
    }'''
    if original not in source:
        raise RuntimeError('Upstream half conversion changed; review the portability patch')
    source = source.replace(original, portable)
    source += '\nextern "C" void assembly_half_decode(const uint16_t* x, float* y, int n) { for(int i=0;i<n;++i)y[i]=h2f(x[i]); }\n'
    PLUGIN.parent.mkdir(parents=True, exist_ok=True)
    cpp = PLUGIN.parent / 'portable_sdf.cc'
    cpp.write_text(source)
    mj = Path(mujoco.__file__).parent
    header = (mj / 'include/mujoco/mjplugin.h').read_text()
    if '#define mjPLUGIN_LIB_INIT __attribute__' in header:
        source = source.replace('mjPLUGIN_LIB_INIT(assemblenist_sdf)', 'mjPLUGIN_LIB_INIT')
        cpp.write_text(source)
    library = list(mj.glob('libmujoco*.dylib' if sys.platform == 'darwin' else 'libmujoco.so*'))[0]
    compiler = shutil.which('clang++' if sys.platform == 'darwin' else 'g++')
    if compiler is None:
        raise FileNotFoundError('A C++17 compiler is required for the native contact plugin')
    subprocess.run([compiler, '-O3', '-std=c++17', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                    '-fPIC', '-I' + str(mj / 'include'), str(cpp), str(library), '-Wl,-rpath,' + str(mj),
                    '-o', str(PLUGIN)], check=True)
    test = ctypes.CDLL(str(PLUGIN))
    f = test.assembly_half_decode
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    bits = np.arange(65536, dtype=np.uint16)
    out = np.empty(65536, dtype=np.float32)
    f(bits.ctypes.data, out.ctypes.data, len(bits))
    expected = bits.view(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected)
    metadata = dict(platform=sys.platform, mujoco=mujoco.__version__,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(), half_values_verified=65536)
    (PLUGIN.parent / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Built contact plugin; verified all 65,536 FP16 encodings.', flush=True)
