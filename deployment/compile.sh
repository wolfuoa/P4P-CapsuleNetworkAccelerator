#!/bin/bash

if [ $1 = zcu102 ]; then
      ARCH=${BASEDIR}ZCU102/arch.json
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Pass zcu102 into args!"
      exit 1
fi

compile() {
      vai_c_tensorflow2 \
            --model           models/quant/quantized_partial_model.h5 \
            --arch            $ARCH \
            --output_dir      compiled_model \
            --net_name        partial_caps
}

compile 2>&1 | tee compile.log

echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"



