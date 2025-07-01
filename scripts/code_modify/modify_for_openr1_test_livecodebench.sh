sed -i '/_process_weights_after_loading(model, model_config, target_device)/a\
            if model_config.quant_type is not None:\
                import sys\
                BIT_ROOT = "/scratch/amlt_code/"\
                sys.path.append(BIT_ROOT)\
                sys.path.append(BIT_ROOT + "/test")\
                sys.path.append(BIT_ROOT + "/quantization")\
                from test_utils import pseudo_quantize_model_weight\
                q_config = {\
                    "zero_point": True,  # by default True\
                    "q_group_size": model_config.group_size,  # whether to use group quantization\
                }\
                logger.info(f"Quantizing model weights with config: quant_type={model_config.quant_type}, bits={model_config.bits}, q_config={q_config}")\
                pseudo_quantize_model_weight(\
                    model, w_bit=model_config.bits, q_config=q_config, quant_type=model_config.quant_type\
                )
' ../../venv_openr1/lib/python3.11/site-packages/vllm/model_executor/model_loader/loader.py

sed -i '/model_impl: Union\[str, ModelImpl\] = ModelImpl.AUTO,/{
N
s/model_impl: Union\[str, ModelImpl\] = ModelImpl.AUTO,\n\s*) -> None:/model_impl: Union[str, ModelImpl] = ModelImpl.AUTO,\
        bits: Optional[int] = 2,\
        group_size: Optional[int] = 64,\
        quant_type: Optional[str] = "int",\
    ) -> None:\n        self.bits = bits\
        self.group_size = group_size\
        self.quant_type = quant_type/
}' ../../venv_openr1/lib/python3.11/site-packages/vllm/config.py

sed -i '/^class EngineArgs:/a\
    bits: Optional[int] = 2\
    group_size: Optional[int] = 64\
    quant_type: Optional[str] = "int"
' ../../venv_openr1/lib/python3.11/site-packages/vllm/engine/arg_utils.py

sed -i '/^[[:space:]]*pretrained: str/a\
    bits: int = 2\
    group_size: int = 64\
    quant_type: str = None
' ../../venv_openr1/lib/python3.11/site-packages/lighteval/models/vllm/vllm_model.py

sed -i '/"seed": config.seed,/a\
            "bits": config.bits,\
            "group_size": config.group_size,\
            "quant_type": config.quant_type,
' ../../venv_openr1/lib/python3.11/site-packages/lighteval/models/vllm/vllm_model.py

sed -i 's/codegen_pass@1:16/codegen_pass@1/g' ../../venv_openr1/lib/python3.11/site-packages/lighteval/tasks/extended/lcb/main.py
