/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#pragma once
#include <QtGui>
#include <QMainWindow>
#include "ui/ui_MainWindowBase.h"
#include "gui_export_api.h"
#include <QString>
#include <qlabel.h>

class RenderWidget;
class OptixRenderer;
class Application;
class Camera;

class MainWindowBase : public QMainWindow, public Ui::MainWindowBase
{
    Q_OBJECT
public:
    GUI_EXPORT_API MainWindowBase(Application& application);
    GUI_EXPORT_API ~MainWindowBase();
    GUI_EXPORT_API virtual void closeEvent(QCloseEvent* event);
    static QString getApplicationStatusString(const Application & application, bool showSeconds = true);

signals:
    void renderRestart();
    void renderStatusToggle();
    //void cameraUpdated();

private slots:
    GUI_EXPORT_API_QT void onSetCameraToDefault();
    GUI_EXPORT_API_QT void onChangeRenderMethodPPM();
    GUI_EXPORT_API_QT void onChangeRenderMethodPT();
    GUI_EXPORT_API_QT void onChangeRenderMethodVCM();
    GUI_EXPORT_API_QT void onConfigureGPUDevices();
    void onOpenSceneFile();
    void onReloadLastScene();
    //void onCameraUpdated();
    void onRunningStatusChanged();

    void onRenderMethodChanged(); 
    void onActionAbout();
    void onRenderStatusToggle();
    void onRenderRestart();
    void onUpdateRunningStatusLabelTimer();
    void onApplicationError(QString);
    void onActionSaveImageBMP();
    void onActionOpenBuiltInScene();
    void onActionOpenBuiltInSceneCornell();
    void onActionOpenBuiltInSceneCornellSmall();
    void onActionOpenBuiltInSceneCornellSmallNoBlocks();
    void onActionOpenBuiltInSceneCornellSmallLargeSphere();
    void onActionOpenBuiltInSceneCornellSmallSmallSpheres();
    void onActionOpenBuiltInSceneCornellSmallUpwardsLight();

private:
    void loadSceneByName( QString &fileName );
    RenderWidget* m_renderWidget;
    //void onChangeRenderMethod();
    Application & m_application;
    QLabel* m_statusbar_renderMethodLabel;
    QLabel* m_statusbar_runningStatusLabel;
    unsigned int m_renderWidth;
    unsigned int m_renderHeight;
    QFileInfo m_lastOpenedSceneFile;
    Camera & m_camera;
};
