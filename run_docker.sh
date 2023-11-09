# Expose the X server on the host.
sudo xhost +local:root

docker run \
  --rm \
  -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e MPLCONFIGDIR=/workspace/mplcache \
  --user $(id -u):$(id -g) \
  --volume $(pwd):/root \
  --volume /nrs/branson/kwaki/jumping_data:/nrsdata \
  --volume $(pwd):/workspace iskwak/calibrate \
  bash
