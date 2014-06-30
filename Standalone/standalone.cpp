/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <iostream>
#include <exception>
#include <QApplication>
#include "gui/MainWindowBase.hxx"
#include "StandaloneApplication.h"
#include "ComputeDeviceRepository.h"
#include <QThread>
#include <QTextStream>
#include <qmessagebox.h>
#include <tchar.h>

//#include <vld.h>


int fileExists(char * file)
{
    WIN32_FIND_DATA FindFileData;
    HANDLE handle = FindFirstFile(file, &FindFileData) ;
    int found = handle != INVALID_HANDLE_VALUE;
    if (found) 
    {
        FindClose(handle);
    }
    return found;
}


void attachConEmu()
{
#ifndef _WIN64
    char * conEmuPath = "C:\\Tools\\ConEmu\\ConEmu\\ConEmuC.exe";
    char * createProcCmd = "\"C:\\Tools\\ConEmu\\ConEmu\\ConEmuC.exe\" /AUTOATTACH";
#else
    char * conEmuPath = "C:\\Tools\\ConEmu\\ConEmu\\ConEmuC64.exe";
    char * createProcCmd = "\"C:\\Tools\\ConEmu\\ConEmu\\ConEmuC64.exe\" /AUTOATTACH";
#endif
    if (fileExists(conEmuPath))
    {
        printf("Attaching ConEmu console\n");
        STARTUPINFO si = { sizeof(si) }; 
        PROCESS_INFORMATION pi = {};
        if (CreateProcess(NULL, createProcCmd,
            NULL, NULL, FALSE, NORMAL_PRIORITY_CLASS, NULL, NULL, &si, &pi))
        { 
            CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
        }
    }
}


int main( int argc, char** argv )
{
    QApplication qApplication(argc, argv);
    qApplication.setOrganizationName("Opposite Renderer");
    qApplication.setApplicationName("Opposite Renderer");
    attachConEmu();

    QTextStream out(stdout);
    QTextStream in(stdin);
    
    setvbuf(stdout, NULL, _IONBF, NULL);	// vmarz: disable output stream buffering, otherwise printf output doesn't show up consistently
    setvbuf(stderr, NULL, _IONBF, NULL);

    try
    {
        ComputeDeviceRepository repository;

        const std::vector<ComputeDevice> & repo = repository.getComputeDevices();

        if(repo.empty())
        {
            out << "You must have a CUDA enabled GPU to run this application." 
                << endl << "Press ENTER to quit." << endl;
            in.read(1);
            return 1;
        }

        out << "Available compute devices:" << endl;

        for(int i = 0; i < repo.size(); i++)
        {
            const ComputeDevice & device = repo.at(i);
            out << "   " <<  i << ": " << device.getName() << " (CC " << device.getComputeCapability() << " PCI Bus "<< device.getPCIBusId() <<")" << endl;
        }

        int deviceNumber = repo.size() == 1 ? 0 : -1;

        while (deviceNumber >= repo.size() || deviceNumber < 0)
        {
            out << "Select 0-" << repo.size()-1 << ":" << endl;
            in >> deviceNumber;
        }

        out << deviceNumber << endl;

        ComputeDevice device = repo.at(deviceNumber);
        StandaloneApplication application = StandaloneApplication(qApplication, device);

        // Run application
        QThread* applicationThread = new QThread(&qApplication);
        application.moveToThread(applicationThread);
        applicationThread->start();

        MainWindowBase mainWindow(application);
        //mainWindow.showMaximized();
        mainWindow.show();
        mainWindow.resize(1150,700);

        // vmarz: start render manager after MainWindow initialization when all signals/slots hooked up
        application.startRenderManager();
        int returnCode = qApplication.exec();
        application.wait();

        applicationThread->quit();
        applicationThread->wait();

        return returnCode;
    }
    catch(const std::exception & E)
    {
        QMessageBox::warning(NULL, "Exception Thrown During Launch of Standalone", E.what());
        return -1;
    }
}
