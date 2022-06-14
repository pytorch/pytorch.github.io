#!/usr/bin/env bash

set -e

# ruby-install --latest

OPENSSL_VERSION=1.1
OPENSSL_DIR=/usr/local/opt/openssl@${OPENSSL_VERSION}

function install-ruby-on-mac() {
    PKG_CONFIG_PATH=${OPENSSL_DIR}/lib/pkgconfig \
    ruby-install $(ruby-install-options $1) \
        ruby $1 \
        -- \
        --with-openssl-dir=${OPENSSL_DIR} \
        --with-opt-dir=$(brew --prefix readline) \
        --without-tcl --without-tk
}

install-ruby-on-mac 2.6.10
