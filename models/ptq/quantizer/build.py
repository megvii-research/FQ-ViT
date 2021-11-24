from .uniform import UniformQuantizer
from .log2 import Log2Quantizer


str2quantizer = {
    "uniform": UniformQuantizer,
    "log2": Log2Quantizer
}


def build_quantizer(quantizer_str, bit_type, observer,
                    module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
