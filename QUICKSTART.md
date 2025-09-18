# PEDS Walkability - Quick Start Guide

## ✅ Setup Complete!

Your environment is ready. Here's how to run the P0 functionality:

## 🚀 Run the App

1. **Set your Google Maps API key:**
   ```bash
   set GOOGLE_MAPS_API_KEY=your_actual_api_key_here
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Use the app:**
   - Click on the map to select a location
   - Configure Street View settings in the sidebar
   - Upload or select a SAM model (.pth file)
   - Click "Fetch & Segment" to analyze

## 📋 What You Need

### Required:
- ✅ **Python environment** - Set up and activated
- ✅ **Dependencies** - All installed
- 🔑 **Google Maps API Key** - [Get one here](https://console.cloud.google.com/apis/credentials)
- 📁 **SAM Model** - Download a .pth file to the `models/` folder

### SAM Model Download:
Download one of these SAM checkpoints to your `models/` folder:
- **vit_h**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth (2.6GB)
- **vit_l**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth (1.2GB)  
- **vit_b**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth (375MB)

## 🔧 Google Maps API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create or select a project
3. Enable billing
4. Create an API key
5. Enable these APIs:
   - Street View Static API
   - Places API (optional, for future features)

## 🎯 P0 Workflow Test

Once running, test this workflow:
1. **Map Selection** - Click anywhere on the map
2. **Street View** - Should fetch and display the street view image
3. **SAM Processing** - Should generate segmentation masks
4. **Results** - Should show overlay and allow mask download

## 🐛 Troubleshooting

- **"Street View not available"** - Try a different location (urban areas work best)
- **"SAM checkpoint not found"** - Download a .pth file to the models/ folder
- **API errors** - Check your API key and billing setup

## 📞 Need Help?

If you encounter issues with the P0 workflow, let me know:
- What error messages you see
- Which step fails
- Your API key setup status