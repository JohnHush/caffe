set(FRCNN_api_sources
  frcnn_api.cpp
  rpn_api.cpp
  )
ADD_LIBRARY(FRCNN_api ${FRCNN_api_sources})
TARGET_LINK_LIBRARIES(FRCNN_api ${Caffe_LINK})

install(DIRECTORY ${Caffe_INCLUDE_DIR}/api DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS FRCNN_api EXPORT CaffeTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
