/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Stian Pedersen
 *                Valdis Vilcans
*/

#include "SceneFactory.h"
#include "scene/IScene.h"
#include "scene/Cornell.h"
#include "scene/CornellSmall.h"

SceneFactory::SceneFactory(void)
{
}


SceneFactory::~SceneFactory(void)
{
}

IScene* SceneFactory::getSceneByName( const char* name )
{
    if (strcmp(name, "Cornell") == 0)
    {
        return new Cornell();
    }
    else if (strcmp(name, "CornellSmall") == 0)
    {
        return new CornellSmall(CornellSmall::Default);
    }
    else if (strcmp(name, "CornellSmallNoBlocks") == 0)
    {
        return new CornellSmall(CornellSmall::LightArea);
    }
    else if (strcmp(name, "CornellSmallLargeSphere") == 0)
    {
        return new CornellSmall(CornellSmall::SmallVCMColors | CornellSmall::BackwallBlue | CornellSmall::FloorGlossy | 
                                CornellSmall::LargeMirrorSphere | CornellSmall::LightArea);
    }
    else if (strcmp(name, "CornellSmallSmallSpheres") == 0)
    {
        return new CornellSmall(CornellSmall::SmallVCMColors | CornellSmall::BackwallBlue | CornellSmall::FloorGlossy |
                                CornellSmall::LightPointStrong | CornellSmall::SmallGlassSphere | CornellSmall::SmallMirrorSphere);
    }
    else if (strcmp(name, "CornellSmallLightUpwards") == 0)
    {
        return new CornellSmall(CornellSmall::SmallVCMColors | CornellSmall::BackwallBlue | CornellSmall::LightAreaUpwards );
    }
    else if (strcmp(name, "CornellSmallPointDistant") == 0)
    {
        return new CornellSmall(CornellSmall::SmallVCMColors | CornellSmall::BackwallBlue | //CornellSmall::FloorGlossy |
            CornellSmall::LightPointDistant | CornellSmall::SmallGlassSphere | CornellSmall::SmallMirrorSphere);
    }
    else if (strcmp(name, "CornellSmallPointTest") == 0)
    {
        return new CornellSmall(CornellSmall::SmallVCMColors | CornellSmall::BackwallBlue | //CornellSmall::FloorGlossy |
            //CornellSmall::SmallMirrorSphere |
            CornellSmall::SmallGlassSphere |
            CornellSmall::FloorGlossy |
            CornellSmall::LightPointStrong);
    }
    else
    {
        return Scene::createFromFile(name);
    }
}
