# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['neurodemo.py'],
    pathex=[],
    binaries=[],
    datas=[('neurodemo/images', 'images')],
    hiddenimports=[
        'pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt6',
        'pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt6',
        'pyqtgraph.imageview.ImageViewTemplate_pyqt6',
        'pyqtgraph.console.template_pyqt6',
        'ctypes',
        'numpy',
        'scipy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='neurodemo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='neurodemo',
)
app = BUNDLE(
    coll,
    name='neurodemo.app',
    icon='icon.icns',
    bundle_identifier='edu.unc.neurodemo',
    info_plist={
        'CFBundleShortVersionString': '1.4',
        'NSHighResolutionCapable': 'True',
    },
)
