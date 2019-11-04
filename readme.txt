在FramePublish.h中定义#define USEKLT，开启klt跟踪模式，关闭该宏为orb跟踪模式。
跟踪模式中有分块补点和全局补点两种模式，在se2lamKltGyroTrack.cpp中设置
system.receiveImuTheta(outImuData.dtheta,ctime, true);//ture时采用分块，false时采用全局