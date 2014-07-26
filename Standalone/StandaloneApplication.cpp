/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#include "StandaloneApplication.h"
#include "ComputeDevice.h"
#include <QApplication>
#include <QThread>

StandaloneApplication::StandaloneApplication(QApplication & qApplication, const ComputeDevice & device)
    : Application(qApplication),
      m_renderManager(StandaloneRenderManager(qApplication, *this, device))
{
    connect(&m_renderManager, SIGNAL(newFrameReadyForDisplay(const float*, const unsigned long long*, QMutex*)), 
            this,             SIGNAL(newFrameReadyForDisplay(const float*, const unsigned long long*, QMutex*)));

    // Run render manager in thread
    m_thread = new QThread(&qApplication);
    m_thread->setObjectName("QThread::StandaloneRenderRenderManager");
    m_renderManager.moveToThread(m_thread);
    QObject::connect(m_thread, SIGNAL(started()), &m_renderManager, SLOT(start()));

    // Pass on render manager errors as application errors
    connect(&m_renderManager, SIGNAL(renderManagerError(QString)), 
        this, SIGNAL(applicationError(QString)),
        Qt::QueuedConnection);
}

StandaloneApplication::~StandaloneApplication(void)
{

}

void StandaloneApplication::startRenderManager()
{
    // vmarz: start renderer manager after rest of the initialization done and all 
    // signals/slots hooked up so it has chance to report initialization errors
    m_thread->start();
}

void StandaloneApplication::wait()
{
    waitOnApplicationFinished();
    m_thread->quit();
    m_thread->wait();
}
