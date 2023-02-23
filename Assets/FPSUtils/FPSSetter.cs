using UnityEngine;

public class FPSSetter : MonoBehaviour
{
    private const int TargetFPS = 3000;

    [RuntimeInitializeOnLoadMethod]
    private static void SetFPS()
    {
        QualitySettings.vSyncCount  = 0;
        Application.targetFrameRate = TargetFPS;
    }
}