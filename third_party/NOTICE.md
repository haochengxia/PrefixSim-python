# Third-party code in the binary wheels

The `prefixsim` wheels contain, besides this package (Apache-2.0):

| Component | License | How | Source |
|---|---|---|---|
| [PrefixSim](https://github.com/cacheMon/PrefixSim) | Apache-2.0 | static | `extern/PrefixSim` |
| [libCacheSim](https://github.com/1a1a11a/libCacheSim) | Apache-2.0 (see its NOTICE) | static | `extern/PrefixSim/libCacheSim` |
| [GLib](https://gitlab.gnome.org/GNOME/glib) 2.68 | LGPL-2.1-or-later, `glib-LGPL-2.1.txt` | static (Linux), bundled dylib (macOS) | AlmaLinux 9 `glib2`, or GNOME's 2.68 release |
| [PCRE](https://www.pcre.org/) 8.44, used by GLib | BSD-3-Clause, `pcre-LICENCE.txt` | static (Linux) | AlmaLinux 9 `pcre` |
| [sysprof-capture](https://gitlab.gnome.org/GNOME/sysprof) 3.40, used by GLib | BSD-2-Clause-Patent, `sysprof-capture-COPYING.txt` | static (Linux) | AlmaLinux 9 `sysprof` |
| [zstd](https://github.com/facebook/zstd) 1.5 | BSD-3-Clause, `zstd-LICENSE.txt` | bundled shared library | AlmaLinux 9 `zstd`, or Homebrew |
| [pybind11](https://github.com/pybind/pybind11) | BSD-3-Clause | headers | PyPI `pybind11` |

GLib is linked statically into `prefixsim/_core` on Linux. As LGPL-2.1 section
6 permits, the complete source of everything else in the module is this
repository (and the sdist on PyPI), so the module can be relinked with a
modified GLib: build it from source against that GLib (see the README).
