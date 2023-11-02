# Expose the X server on the host.
sudo xhost +local:root

# --rm: Make the container ephemeral (delete on exit).
# -it: Interactive TTY.
# --gpus all: Expose all GPUs to the container.
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
  --volume $(pwd):/workspace ikwak/calibrate \
  bash
# docker run \
#   --rm \
#   -it \
#   --gpus all \
#   -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -e DISPLAY=$DISPLAY \
#   -e QT_X11_NO_MITSHM=1 \
#   -e MPLCONFIGDIR=/workspace/mplcache \
#   --user $(id -u):$(id -g) \
#   --volume $(pwd):/root \
#   --volume /nrs/branson/kwaki/jumping_data:/nrsdata \
#   --volume $(pwd):/workspace kwaki/cudagl \
#   bash
