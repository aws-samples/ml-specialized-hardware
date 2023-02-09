# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# install required packages
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
    libchromaprint1 libgles2 libgmp10 libgme0 libxkbcommon0  \
    libbs2b0 libgsl27 libwavpack1 libxslt1.1 gstreamer1.0-libav  \
    libavformat58 libwayland-server0 libwayland-client0 libharfbuzz-icu0 librtmp1  \
    libtheora0 libxtst6 libmpg123-0 libxcomposite1 libgl1  \
    libmjpegutils-2.1-0 python3-numpy libsrt1.4-openssl libmpeg2-4 libvulkan1  \
    libxdamage1 libjpeg8 mjpegtools libpng16-16 wayland-protocols  \
    libcap2 libofa0 udev libgstreamer-plugins-base1.0-dev libcups2  \
    libgstreamer1.0-dev libopenexr25 libmfx1 libde265-0 libgirepository1.0-dev  \
    libfdk-aac2 libavcodec58 git libunwind8 xdg-dbus-proxy  \
    libtwolame0 mesa-utils libtag1v5 libaa1 libgles1  \
    ffmpeg liborc-0.4-0 libgraphene-1.0-dev libwebpdemux2 libsoup2.4-1  \
    build-essential libsm6 libglu1 libwebrtc-audio-processing1 liba52-0.7.4  \
    libva2 libwayland-cursor0 libcurl3-gnutls libvisual-0.4-0 libbz2-1.0  \
    libvpx7 libdv4 libatspi2.0-0 liblilv-0-0 gstreamer1.0-plugins-bad  \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly libdvdnav4 libssl3 libgsm1  \
    libwoff1 libwebp7 x264 libopenblas-dev libxrandr-dev  \
    freeglut3-dev intel-media-va-driver-non-free ladspa-sdk gfortran libvo-aacenc0  \
    python3-opencv libcaca0 python3-opengl libsbc1 libatk1.0-0  \
    libsoundtouch1 libsndfile1 python3 libgudev-1.0-0 liblcms2-2  \
    libzvbi0 libatk-bridge2.0-0 libass9 libgbm1 libglib2.0-0  \
    libaom3 bubblewrap libdw1 libseccomp2 libepoxy0  \
    libavutil56 glibc-tools libmodplug1 libshout3 libwebpmux3  \
    libfaad2 iso-codes libgcrypt20 xvfb libspandsp2  \
    libvorbis0a libfaac0 libmpcdec6 libopus0 libsrtp2-1  \
    libx264-163 gstreamer1.0-rtsp python3-pip ca-certificates libva-wayland2  \
    gcc libwildmidi2 libpango-1.0-0 libflite1 libdca0  \
    libopenjp2-7 libzbar0 libspeex1 libkate1 pkg-config  \
    libx264-dev libopencore-amrwb0 gstreamer1.0-tools libxv1 gstreamer1.0-plugins-base  \
    libcairo2-dev python3-gst-1.0 wget cmake libwayland-egl1  \
    libavfilter7 libegl1 libdvdread8 libvo-amrwbenc0 libogg0  \
    librsvg2-2 libopencore-amrnb0 libx265-199 libatk-adaptor sudo  \
    libmp3lame0 python3-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# download and install libjpeg turbo
RUN wget -qO libjpeg-turbo.deb \
    https://deac-ams.dl.sourceforge.net/project/libjpeg-turbo/2.1.4/libjpeg-turbo-official_2.1.4_amd64.deb && \
    dpkg -i libjpeg-turbo.deb && \
    rm -f libjpeg-turbo.deb

# download and install aws plugins for gstreamer
RUN wget -qO sh.rustup.rs https://sh.rustup.rs && \
    bash sh.rustup.rs -q -y --profile default && \
    . "$HOME/.cargo/env" && \
    rm -f sh.rustup.rs && \
    cargo install cargo-c && \
    git clone -b gstreamer-1.21.1 https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git && \
    cd gst-plugins-rs && \
    cargo cbuild -p gst-plugin-aws --libdir=/usr/lib/x86_64-linux-gnu && \
    cargo cinstall -p gst-plugin-aws --libdir=/usr/lib/x86_64-linux-gnu && \
    cd .. && rm -rf gst-plugins-rs

# create a user
RUN mkdir -p /opt/ml/processing/output /opt/ml/processing/code
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN groupadd --gid 500 --non-unique ec2-user
RUN adduser --uid 500 --disabled-password --gecos '' --ingroup ec2-user ec2-user
RUN usermod -a -G sudo,video,ec2-user ec2-user
ENV PATH="$PATH:/home/ec2-user/.local/bin"
RUN chown -R ec2-user:ec2-user /opt/ml

USER ec2-user
WORKDIR /opt/ml/processing/code
# install some required python packages
RUN pip3 install --upgrade pip
RUN pip3 install pycairo PyGObject PyTurboJPEG boto3 Cython
# torch is required for ByteTrack
RUN pip3 install torch torchvision thop loguru scikit-learn lap cython_bbox

RUN echo "#!/bin/sh\n/usr/bin/xvfb-run -a \$@\n" > /home/ec2-user/entrypoint.sh && chmod +x /home/ec2-user/entrypoint.sh


ENTRYPOINT [ "/home/ec2-user/entrypoint.sh"]
