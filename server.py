diff --git a/server.py b/server.py
index 1b2c3d4..5e6f7g8 100644
--- a/server.py
+++ b/server.py
@@ -100,7 +100,7 @@ def predict_aqi(...)
    # Calculate the consensus AQI value
-   consensus_aqi = (rf_prediction + svr_prediction + gbr_prediction) / 2.0
+   consensus_aqi = (rf_prediction + svr_prediction + gbr_prediction) / 3.0

    return consensus_aqi
