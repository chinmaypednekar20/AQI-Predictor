diff --git a/app.py b/app.py
index 1234567..89abcde 100644
--- a/app.py
+++ b/app.py
@@ -20,10 +20,10 @@ def predict_air_quality(data):
     features = {
         'Feature1': data['Feature1'],
         'Feature2': data['Feature2'],
-        'PM25': data['PM25'],
+        'PM2.5': data['PM25'],
         'Feature3': data['Feature3'],
     }
     return model.predict(features)
