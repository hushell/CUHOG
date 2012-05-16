VERSION = 0.7.0
TEMPLATE = lib
CONFIG += debug console

QMAKE_CXXFLAGS += -fPIC -ffast-math -fomit-frame-pointer 
QMAKE_CFLAGS += -fPIC -ffast-math -fomit-frame-pointer 

HEADERS = get_cells.h get_features.h voc_hog.h process.h timer.h global.h \
HOGDefines.h HOGUtils.h
SOURCES = voc_hog.cpp process.cpp
CUSOURCES = get_cells.cu get_features.cu timer.cu HOGUtils.cu

CUDA_SDK_PATH = /home/hushell/NVIDIA_GPU_Computing_SDK
LIBS += -lcudart -L/usr/local/cuda/lib64 -L$${CUDA_SDK_PATH}/C/lib -lcutil_x86_64 -lrt

QMAKE_CUC = nvcc
cu.name = Cuda ${QMAKE_FILE_IN}
cu.input = CUSOURCES
cu.CONFIG += no_link
cu.variable_out = OBJECTS

INCLUDEPATH += /home/hushell/NVIDIA_GPU_Computing_SDK/C/common/inc
INCLUDEPATH += /usr/local/cuda/include 
QMAKE_CUFLAGS += $$QMAKE_CFLAGS
## QMAKE_CUEXTRAFLAGS += -arch=sm_20 --ptxas-options=-v -Xcompiler -fPIC -Xcompiler $$join(QMAKE_CUFLAGS, ",")
QMAKE_CUEXTRAFLAGS += -arch=sm_20 -Xcompiler -fPIC -Xcompiler $$join(QMAKE_CUFLAGS, ",")
QMAKE_CUEXTRAFLAGS += $(DEFINES) $(INCPATH) $$join(QMAKE_COMPILER_DEFINES, " -D", -D)
QMAKE_CUEXTRAFLAGS += -c

cu.commands = $$QMAKE_CUC $$QMAKE_CUEXTRAFLAGS -o ${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} ${QMAKE_FILE_NAME}$$escape_expand(\n\t)
cu.output = ${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
silent:cu.commands = @echo nvcc ${QMAKE_FILE_IN} && $$cu.commands
QMAKE_EXTRA_COMPILERS += cu

build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
else:cuclean.CONFIG += recursive
QMAKE_EXTRA_TARGETS += cuclean
