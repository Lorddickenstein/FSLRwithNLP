# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['D:/Documents/Thesis/FSLRwithNLP/Application/FslrApplication.py'],
             pathex=['D:/Documents/Thesis/FSLRwithNLP/Application'],
             binaries=[],
             datas=[('C:/Users/jerso/AppData/Local/Programs/Python/Python39/Lib/site-packages/mediapipe','mediapipe'), ('D:/Documents/Thesis/FSLRwithNLP/Application/Figures', 'Figures/'), ('D:/Documents/Thesis/FSLRwithNLP/Application/Keyframes', 'Keyframes/'), ('D:/Documents/Thesis/FSLRwithNLP/Application/Models', 'Models/'), ('D:/Documents/Thesis/FSLRwithNLP/Application/NLP', 'NLP/'), ('D:/Documents/Thesis/FSLRwithNLP/Application/__init__.py', '.'), ('D:/Documents/Thesis/FSLRwithNLP/Application/Images', 'Images/'), ('D:/Documents/Thesis/FSLRwithNLP/Application/evaluator.py', '.'), ('D:/Documents/Thesis/FSLRwithNLP/Application/FSLInterpreter.bat', '.'), ('D:/Documents/Thesis/FSLRwithNLP/Application/HandTrackingModule.py', '.'), ('D:/Documents/Thesis/FSLRwithNLP/Application/FslrApplication.py', '.'), ('D:/Documents/Thesis/FSLRwithNLP/Application/SignClassificationModule.py', '.'), ('D:/Documents/Thesis/FSLRwithNLP/Application/utils.py', '.')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='FslrApplication',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
		  icon='Images/logo.ico',
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='FslrApplication')
