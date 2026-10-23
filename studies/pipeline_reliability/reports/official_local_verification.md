# 已核实结论

官方修订目录62/62份JSON与本机选用索引byte、semantic、size全部一致。独立物理路径检查和freshhash证明所列SG2/SG3同内容对是不同文件；作者原始图片压缩包尚未认证，不将本机像素副本直接归咎官方数据声明。下面保留完整核验过程与早期下载配额阻碍。

Independent official/local verification

Official checkout: ce808160e0df31544b41c7b60b942849ba88bca8. download_JSON.txt is empty; README supplies Drive folder links. Public listing accessible, but all attempted JSON downloads returned quota-exceeded HTML, including exact StyleGAN2 FF file ID provided by parent. Official/local byte or semantic equality remains unverified.

60 independently checked local image samples matched direct physical interpretation, current resolver, and fresh byte/RGB hashes. Resolver has only method-scoped CDF component aliases and explicit rddm/RDDM casing; no general cross-method fallback.

Local FF StyleGAN2/3 have identical unique byte and RGB sets: train 15586, test 3591. Sampled method pairs have distinct realpaths and inodes. This demonstrates actual separate local files, rather than two index methods resolving to one file. It does not prove original author archives have identical files.

Cross-role fresh checks:
- StyleGAN2 seed19218: 2 files; byte equality True; RGB equality True; paths/inodes preserved in JSON.
- StyleGAN2 seed10143: 2 files; byte equality True; RGB equality True; paths/inodes preserved in JSON.
- StyleGAN2 seed18935: 2 files; byte equality True; RGB equality True; paths/inodes preserved in JSON.
- StyleGAN3 seed19218: 2 files; byte equality True; RGB equality True; paths/inodes preserved in JSON.
- StyleGAN3 seed10143: 2 files; byte equality True; RGB equality True; paths/inodes preserved in JSON.
- StyleGAN3 seed18935: 2 files; byte equality True; RGB equality True; paths/inodes preserved in JSON.

No roles, dataset contents, source code, GPU state, or training state changed.

Author-revised StyleGAN2 FF file ID 1AYWN4UMeqOve8L-tBOrflkIqhSyjupIm also returned quota-exceeded HTML. Failed responses were renamed .quota_response.html so they cannot be mistaken for authentic JSON metadata; response sizes and SHA256 retained.

Update: connected Drive raw fetch preserved complete revised StyleGAN2/3 FF JSON (5,561,395 bytes each). Full role/path comparison results and exact byte hashes are in official_local_verification.json; earlier direct-download quota blockage remains provenance only.
[
  {
    "method": "StyleGAN2",
    "file_id": "1AYWN4UMeqOve8L-tBOrflkIqhSyjupIm",
    "modified_time": "2024-10-24T11:34:15.000Z",
    "official_bytes": 5561395,
    "official_sha256": "3e255c61e37721b43e69dbc5052eb8356e1b59f3786ad23ae610a42c508ce189",
    "local_sha256": "3e255c61e37721b43e69dbc5052eb8356e1b59f3786ad23ae610a42c508ce189",
    "byte_equal": true,
    "semantic_equal": true,
    "roles": {
      "StyleGAN2_Fake|test": {
        "official": {
          "groups": 116,
          "nonempty": 116,
          "frames": 3711,
          "unique_paths": 3711
        },
        "local": {
          "groups": 116,
          "nonempty": 116,
          "frames": 3711,
          "unique_paths": 3711
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      },
      "StyleGAN2_Fake|train": {
        "official": {
          "groups": 564,
          "nonempty": 564,
          "frames": 18007,
          "unique_paths": 18007
        },
        "local": {
          "groups": 564,
          "nonempty": 564,
          "frames": 18007,
          "unique_paths": 18007
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      },
      "StyleGAN2_Real|test": {
        "official": {
          "groups": 140,
          "nonempty": 140,
          "frames": 4479,
          "unique_paths": 4479
        },
        "local": {
          "groups": 140,
          "nonempty": 140,
          "frames": 4479,
          "unique_paths": 4479
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      },
      "StyleGAN2_Real|train": {
        "official": {
          "groups": 719,
          "nonempty": 719,
          "frames": 22993,
          "unique_paths": 22993
        },
        "local": {
          "groups": 719,
          "nonempty": 719,
          "frames": 22993,
          "unique_paths": 22993
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      }
    }
  },
  {
    "method": "StyleGAN3",
    "file_id": "1MTt8lso3M5xKBqYpNaUMgKZI4QCC_vSN",
    "modified_time": "2024-10-24T11:34:15.000Z",
    "official_bytes": 5561395,
    "official_sha256": "941f9914489383fb83b5422aaaefefcda8830a75dbf50e103b7171214a345e18",
    "local_sha256": "941f9914489383fb83b5422aaaefefcda8830a75dbf50e103b7171214a345e18",
    "byte_equal": true,
    "semantic_equal": true,
    "roles": {
      "StyleGAN3_Fake|test": {
        "official": {
          "groups": 116,
          "nonempty": 116,
          "frames": 3711,
          "unique_paths": 3711
        },
        "local": {
          "groups": 116,
          "nonempty": 116,
          "frames": 3711,
          "unique_paths": 3711
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      },
      "StyleGAN3_Fake|train": {
        "official": {
          "groups": 564,
          "nonempty": 564,
          "frames": 18007,
          "unique_paths": 18007
        },
        "local": {
          "groups": 564,
          "nonempty": 564,
          "frames": 18007,
          "unique_paths": 18007
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      },
      "StyleGAN3_Real|test": {
        "official": {
          "groups": 140,
          "nonempty": 140,
          "frames": 4479,
          "unique_paths": 4479
        },
        "local": {
          "groups": 140,
          "nonempty": 140,
          "frames": 4479,
          "unique_paths": 4479
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      },
      "StyleGAN3_Real|train": {
        "official": {
          "groups": 719,
          "nonempty": 719,
          "frames": 22993,
          "unique_paths": 22993
        },
        "local": {
          "groups": 719,
          "nonempty": 719,
          "frames": 22993,
          "unique_paths": 22993
        },
        "path_sets_equal": true,
        "path_multisets_equal": true,
        "official_only": 0,
        "local_only": 0
      }
    }
  }
]

Final metadata verification: 62/62 byte-identical and 62/62 semantically identical to local selected indices; 62/62 sizes match official metadata. All original official JSON bytes preserved under artifacts/data/official_verification/official_metadata. No metadata reconstruction or role reinterpretation was used. Author archive image bytes remain unverified.
