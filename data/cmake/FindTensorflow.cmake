execute_process(COMMAND python3 -c "import sys, os; sys.stdout = open(os.devnull, 'w'); import tensorflow as tf; sys.stdout = sys.__stdout__; print(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import sys, os; sys.stdout = open(os.devnull, 'w'); import tensorflow as tf; sys.stdout = sys.__stdout__; print(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

find_path(TENSORFLOW_INCLUDE_DIR tensorflow/core/public/session.h HINTS ${TF_INCLUDE_DIR} NO_DEFAULT_PATH)
find_library(TENSORFLOW_LIBRARY libtensorflow_framework.so.1 HINTS ${TF_LIB_DIR} NO_DEFAULT_PATH)

set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIR} "${TENSORFLOW_INCLUDE_DIR}/external/protobuf_archive/src/")
set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tensorflow DEFAULT_MSG TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)

mark_as_advanced(TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)
