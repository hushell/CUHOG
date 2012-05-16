TEMPLATE = app
QT = gui
CONFIG += debug console

HEADERS =
SOURCES = main.cpp

INCLUDEPATH += ../../src 
LIBS += -lVOCHOG -L../../src -L/usr/local/cuda/lib64 

DESTDIR = ../../bin
