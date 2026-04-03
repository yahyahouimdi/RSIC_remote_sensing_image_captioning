# 🚀 Quick Setup Guide for Image Captioning API

## For Your Teammates - Get Started in 5 Minutes

### Step 1: Get the Files
Ask you for these files:
```
fastapi_service.py
requirements.txt
client_example.py
README_API.md
best_model_improved.pth  (the trained model file)
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start the API Service
**Run this command in your project folder:**
```bash
python fastapi_service.py
```

You should see:
```
Starting Image Captioning FastAPI Service...
Service will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs
```

✅ **The service is ready!**

---

## 🎯 Using the API in Your Project

### Option 1: Call from Python
```python
import requests

# Upload an image and get a caption
image_path = "satellite_image.jpg"

with open(image_path, 'rb') as f:
    response = requests.post(
        "http://localhost:8000/caption/greedy",
        files={'file': f}
    )
    result = response.json()
    caption = result['caption']
    print(f"Generated: {caption}")
```

### Option 2: Call from JavaScript/Frontend
```javascript
const formData = new FormData();
const fileInput = document.getElementById('imageInput');
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/caption/greedy', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => {
    console.log('Caption:', data.caption);
});
```

### Option 3: Call from cURL (Testing)
```bash
curl -X POST "http://localhost:8000/caption/greedy" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

---

## 📊 API Endpoints

| Endpoint | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| `/caption/greedy` | ⚡ Fast | Good | Real-time applications |
| `/caption/beam` | 🏃 Slower | Better | Offline batch processing |
| `/caption/compare` | 🐢 Slowest | Compare | Finding the best caption |

---

## 🧪 Test It Immediately

```bash
# In the project folder, run:
python client_example.py
```

This will test the API if you have a sample image in the folder.

---

## 🌐 API Documentation

Once the service is running, visit:
- **Interactive Docs**: `http://localhost:8000/docs`
- **Try it out**: Click "Try it out" on any endpoint
- **Schema Explorer**: `http://localhost:8000/redoc`

---

## ⚠️ Troubleshooting

**"Cannot connect to http://localhost:8000"**
- Make sure `python fastapi_service.py` is still running

**"Model not loaded"**
- Check that `best_model_improved.pth` is in the same folder

**"ModuleNotFoundError"**
- Run: `pip install -r requirements.txt`

**"Address already in use"**
- Change port in fastapi_service.py line 365: `--port 5000`

---

## 🔗 Sharing the Service Network-Wide

Want to use the API from another computer?

**On the service machine:**
```bash
python fastapi_service.py
# Note your IP: ipconfig (Windows) or ifconfig (Mac/Linux)
# e.g., 192.168.1.100
```

**On client machines:**
```python
# Replace localhost with your IP
requests.post("http://192.168.1.100:8000/caption/greedy", files=files)
```

---

## 📝 Integration Checklist

- [ ] Requirements installed: `pip install -r requirements.txt`
- [ ] Service running: `python fastapi_service.py`
- [ ] Model file exists: `best_model_improved.pth`
- [ ] API responding: Open `http://localhost:8000/docs`
- [ ] Integration tested: Call API from your code
- [ ] Share results with team!

---

## 💡 Pro Tips

1. **Batch Processing**: Send multiple images without restarting
2. **Error Handling**: Always check `result.get('success')` in responses
3. **Image Size**: Works best with images 224x224 or larger
4. **Performance**: GPU makes it ~10x faster

---

## 📞 Need Help?

1. Check the full **README_API.md** for advanced usage
2. Visit API docs at **http://localhost:8000/docs**
3. Run **client_example.py** to see working examples
4. Check **fastapi_service.py** comments for parameter details

---

**That's it! You're ready to integrate image captioning into your platform! 🎉**
