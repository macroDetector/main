# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[('app/models/weights/*', 'app/models/weights')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # torchvision과 학습/배포에 불필요한 대형 패키지들을 여기에 추가합니다.
    excludes=[
        'torchvision', 
        'torchaudio', 
        'numpy.random._examples'
    ],
    noarchive=False,
    optimize=0,
)

# PYZ 단계에서 불필요한 모듈이 포함되지 않도록 정리됩니다.
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True, # UPX가 설치되어 있다면 파일 압축에 도움이 됩니다.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)