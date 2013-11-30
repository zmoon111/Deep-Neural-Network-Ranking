
BLAS_INCLUDE_PATH := ./atlas/include
BLAS_LIB_PATH := /mnt/mmt/dudalong/bin/gotoblas2/lib
BLAS_LIB_PATH := /mnt/mmt/dudalong/bin/atlas-3.0.10/lib
BLAS_LIB_PATH := ./atlas/lib

vpath %.h ./include:./include/common
vpath %.cpp ./src:./src/common

INCLUDE	=	-I ./include \
			-I ./include/common \
			-I $(BLAS_INCLUDE_PATH)

LIBRARY	= 	-L./ \
			-L$(BLAS_LIB_PATH) \
			-lcdnn-score \
			-lcrypto -lm -lz -lpthread \
			-lcblas -latlas

CC = g++
CPPFLAGS = -g -Wall -fPIC -D_REENTRANT -DDEBUG_ON -DFILE_DUMP_ON -O3


all : main main-thread libcdnn-score.a output
main-thread : main-thread.o libcdnn-score.a
	$(CC) -o $@ $^ $(INCLUDE) $(LIBRARY) -DTEST_MAIN $(CPPFLAGS)
main : main.o libcdnn-score.a
	$(CC) -o $@ $^ $(INCLUDE) $(LIBRARY) -DTEST_MAIN $(CPPFLAGS)
libcdnn-score.a : cdnn_score.o convnet.o conv_util.o  data.o layer.o neuron.o util.o matrix.o matrix_ssemul.o
	ar rcv $@ $^
%.o : %.cpp
	$(CC) -c -o $@ $^ $(CPPFLAGS) $(INCLUDE) $(LIBRARY)


OUTPUT_DIR = ./output
LIB_DIR = $(OUTPUT_DIR)/lib
INCLUDE_DIR = $(OUTPUT_DIR)/include
TOOL_DIR = $(OUTPUT_DIR)/tool
output :
	if [ ! -d $(OUTPUT_DIR) ] ; then mkdir $(OUTPUT_DIR) ; fi
	if [ ! -d $(LIB_DIR) ] ; then mkdir $(LIB_DIR) ; fi
	if [ ! -d $(INCLUDE_DIR) ] ; then mkdir $(INCLUDE_DIR) ; fi
	if [ ! -d $(TOOL_DIR) ] ; then mkdir $(TOOL_DIR) ; fi
	find . -maxdepth 3 -name "*.o" -exec rm {} \;
	cp *.a $(LIB_DIR)
	cp atlas/lib/libatlas.a $(LIB_DIR)
	cp atlas/lib/libcblas.a $(LIB_DIR)
	cp atlas/include/*.h $(INCLUDE_DIR)
	cp include/cdnn_score.h $(INCLUDE_DIR)
	cp model-trans/*.py $(TOOL_DIR)
	cp main main-thread $(TOOL_DIR)
	cp src/main.cpp src/main-thread.cpp src/mainso.cpp $(TOOL_DIR)

.PHONY : clean
clean : 
	find . -maxdepth 1 -name "*.o"  -exec rm {} \;
	find . -maxdepth 1 -name "main" -exec rm {} \;
	find . -maxdepth 1 -name "main-thread" -exec rm {} \;
	find . -maxdepth 1 -name "*.a" -exec rm {} \;
	if [ -d $(OUTPUT_DIR) ] ; then rm -r $(OUTPUT_DIR); fi
