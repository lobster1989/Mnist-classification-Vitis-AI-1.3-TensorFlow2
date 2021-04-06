
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
OUTDIR=./compiled_model/zcu104
NET_NAME=customcnn
MODEL=./models/quantized_model.h5

echo "-----------------------------------------"
echo "COMPILING MODEL FOR ZCU104.."
echo "-----------------------------------------"

compile() {
      vai_c_tensorflow2 \
            --model           $MODEL \
            --arch            $ARCH \
            --output_dir      $OUTDIR \
            --net_name        $NET_NAME
	}


compile 2>&1 | tee compile.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
