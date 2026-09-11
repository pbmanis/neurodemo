# make Apple Silicon app
#
rm -rf build_m1 dist_m1
uv run --extra dev pyinstaller setup_m1.spec --distpath dist_m1 --workpath build_m1
mkdir -p dist_m1/dmg

cp -r "dist_m1/neurodemo.app" dist_m1/dmg
# If the DMG already exists, delete it.
test -f "dist_m1/neurodemo_M1.dmg" && rm "dist_m1/neurodemo_M1.dmg"
# get create-dmg from homebrew
create-dmg \
--volname "neurodemo" \
--volicon "icon.icns" \
--window-pos 300 300 \
--window-size 400 300 \
--icon-size 32 \
--icon "neurodemo.app" 100 100 \
--hide-extension "neurodemo.app" \
--app-drop-link 250 100 \
"dist_m1/neurodemo_M1.dmg" \
"dist_m1/dmg/"
