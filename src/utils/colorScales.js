/**
 * Color scale functions for different dataset types
 * Each function takes raw pixel values and returns an rgba color string
 */

export const colorScales = {
  modis_ndvi: (values) => {
    const ndvi = values[0]
    if (ndvi === null || ndvi === undefined || ndvi < -3000) return null
    
    const scaled = ndvi * 0.0001 // MODIS scale factor
    
    if (scaled < 0) return 'rgba(180, 200, 220, 0.9)' // Water/barren
    if (scaled < 0.2) return 'rgba(210, 180, 140, 0.9)' // Very sparse
    if (scaled < 0.4) return 'rgba(200, 200, 100, 0.9)' // Sparse
    if (scaled < 0.6) return 'rgba(120, 200, 100, 0.9)' // Moderate
    if (scaled < 0.8) return 'rgba(50, 180, 50, 0.9)' // Dense
    return 'rgba(20, 120, 20, 0.9)' // Very dense
  },

  modis_evi: (values) => {
    const evi = values[0]
    if (evi === null || evi === undefined || evi < -3000) return null
    
    const scaled = evi * 0.0001
    
    if (scaled < 0) return 'rgba(200, 180, 160, 0.9)'
    if (scaled < 0.2) return 'rgba(220, 200, 120, 0.9)'
    if (scaled < 0.4) return 'rgba(180, 220, 100, 0.9)'
    if (scaled < 0.6) return 'rgba(100, 200, 100, 0.9)'
    if (scaled < 0.8) return 'rgba(40, 160, 60, 0.9)'
    return 'rgba(0, 100, 30, 0.9)'
  },

  chirps_precip: (values) => {
    const precip = values[0]
    if (precip === null || precip === undefined || precip < 0) return null
    
    // Precipitation in mm
    if (precip < 10) return 'rgba(255, 250, 200, 0.9)' // Very dry
    if (precip < 30) return 'rgba(200, 230, 255, 0.9)' // Light
    if (precip < 60) return 'rgba(120, 180, 255, 0.9)' // Moderate
    if (precip < 100) return 'rgba(60, 120, 230, 0.9)' // Heavy
    if (precip < 200) return 'rgba(30, 60, 180, 0.9)' // Very heavy
    return 'rgba(10, 20, 100, 0.9)' // Extreme
  },

  era5_temp2m: (values) => {
    const tempK = values[0]
    if (tempK === null || tempK === undefined || tempK < 200) return null
    
    const tempC = tempK - 273.15
    
    if (tempC < 0) return 'rgba(150, 150, 255, 0.9)' // Freezing
    if (tempC < 10) return 'rgba(100, 200, 255, 0.9)' // Cold
    if (tempC < 20) return 'rgba(100, 255, 200, 0.9)' // Cool
    if (tempC < 25) return 'rgba(200, 255, 100, 0.9)' // Mild
    if (tempC < 30) return 'rgba(255, 200, 100, 0.9)' // Warm
    if (tempC < 35) return 'rgba(255, 150, 80, 0.9)' // Hot
    return 'rgba(200, 50, 50, 0.9)' // Very hot
  },

  era5_soil_moisture: (values) => {
    const sm = values[0]
    if (sm === null || sm === undefined || sm < 0) return null
    
    // Soil moisture in m³/m³
    if (sm < 0.1) return 'rgba(139, 90, 43, 0.9)' // Very dry
    if (sm < 0.2) return 'rgba(210, 180, 140, 0.9)' // Dry
    if (sm < 0.3) return 'rgba(200, 220, 180, 0.9)' // Moderate
    if (sm < 0.4) return 'rgba(150, 200, 220, 0.9)' // Moist
    if (sm < 0.5) return 'rgba(100, 150, 200, 0.9)' // Wet
    return 'rgba(50, 100, 180, 0.9)' // Saturated
  },

  era5_vpd: (values) => {
    const vpd = values[0]
    if (vpd === null || vpd === undefined || vpd < 0) return null
    
    // VPD in kPa
    if (vpd < 0.5) return 'rgba(100, 150, 255, 0.9)' // Low stress
    if (vpd < 1.0) return 'rgba(150, 200, 200, 0.9)' // Slight
    if (vpd < 1.5) return 'rgba(200, 220, 150, 0.9)' // Moderate
    if (vpd < 2.0) return 'rgba(255, 200, 100, 0.9)' // High
    if (vpd < 3.0) return 'rgba(255, 150, 80, 0.9)' // Very high
    return 'rgba(200, 80, 50, 0.9)' // Extreme
  },

  modis_lst_day: (values) => {
    const lstRaw = values[0]
    if (lstRaw === null || lstRaw === undefined || lstRaw < 0) return null
    
    const tempK = lstRaw * 0.02 // MODIS scale factor
    const tempC = tempK - 273.15
    
    if (tempC < 5) return 'rgba(150, 150, 255, 0.9)'
    if (tempC < 15) return 'rgba(100, 200, 255, 0.9)'
    if (tempC < 25) return 'rgba(150, 255, 150, 0.9)'
    if (tempC < 35) return 'rgba(255, 255, 100, 0.9)'
    if (tempC < 45) return 'rgba(255, 150, 80, 0.9)'
    return 'rgba(200, 50, 50, 0.9)'
  },

  modis_lst_night: (values) => {
    const lstRaw = values[0]
    if (lstRaw === null || lstRaw === undefined || lstRaw < 0) return null
    
    const tempK = lstRaw * 0.02
    const tempC = tempK - 273.15
    
    if (tempC < 0) return 'rgba(150, 150, 255, 0.9)'
    if (tempC < 10) return 'rgba(120, 180, 255, 0.9)'
    if (tempC < 15) return 'rgba(150, 200, 200, 0.9)'
    if (tempC < 20) return 'rgba(200, 220, 180, 0.9)'
    if (tempC < 25) return 'rgba(255, 220, 150, 0.9)'
    return 'rgba(255, 180, 120, 0.9)'
  },
}

export const legends = {
  modis_ndvi: [
    { color: 'rgb(20, 120, 20)', label: 'Dense vegetation (> 0.8)' },
    { color: 'rgb(50, 180, 50)', label: 'Moderate-dense (0.6-0.8)' },
    { color: 'rgb(120, 200, 100)', label: 'Moderate (0.4-0.6)' },
    { color: 'rgb(200, 200, 100)', label: 'Sparse (0.2-0.4)' },
    { color: 'rgb(210, 180, 140)', label: 'Very sparse (0-0.2)' },
    { color: 'rgb(180, 200, 220)', label: 'Water/Barren (< 0)' },
  ],

  modis_evi: [
    { color: 'rgb(0, 100, 30)', label: 'Very dense (> 0.8)' },
    { color: 'rgb(40, 160, 60)', label: 'Dense (0.6-0.8)' },
    { color: 'rgb(100, 200, 100)', label: 'Moderate (0.4-0.6)' },
    { color: 'rgb(180, 220, 100)', label: 'Sparse (0.2-0.4)' },
    { color: 'rgb(220, 200, 120)', label: 'Very sparse (0-0.2)' },
    { color: 'rgb(200, 180, 160)', label: 'Barren (< 0)' },
  ],

  chirps_precip: [
    { color: 'rgb(10, 20, 100)', label: 'Extreme (> 200mm)' },
    { color: 'rgb(30, 60, 180)', label: 'Very heavy (100-200mm)' },
    { color: 'rgb(60, 120, 230)', label: 'Heavy (60-100mm)' },
    { color: 'rgb(120, 180, 255)', label: 'Moderate (30-60mm)' },
    { color: 'rgb(200, 230, 255)', label: 'Light (10-30mm)' },
    { color: 'rgb(255, 250, 200)', label: 'Very dry (< 10mm)' },
  ],

  era5_temp2m: [
    { color: 'rgb(200, 50, 50)', label: 'Very hot (> 35°C)' },
    { color: 'rgb(255, 150, 80)', label: 'Hot (30-35°C)' },
    { color: 'rgb(255, 200, 100)', label: 'Warm (25-30°C)' },
    { color: 'rgb(200, 255, 100)', label: 'Mild (20-25°C)' },
    { color: 'rgb(100, 255, 200)', label: 'Cool (10-20°C)' },
    { color: 'rgb(100, 200, 255)', label: 'Cold (0-10°C)' },
    { color: 'rgb(150, 150, 255)', label: 'Freezing (< 0°C)' },
  ],

  era5_soil_moisture: [
    { color: 'rgb(50, 100, 180)', label: 'Saturated (> 0.5)' },
    { color: 'rgb(100, 150, 200)', label: 'Wet (0.4-0.5)' },
    { color: 'rgb(150, 200, 220)', label: 'Moist (0.3-0.4)' },
    { color: 'rgb(200, 220, 180)', label: 'Moderate (0.2-0.3)' },
    { color: 'rgb(210, 180, 140)', label: 'Dry (0.1-0.2)' },
    { color: 'rgb(139, 90, 43)', label: 'Very dry (< 0.1)' },
  ],

  era5_vpd: [
    { color: 'rgb(200, 80, 50)', label: 'Extreme (> 3.0 kPa)' },
    { color: 'rgb(255, 150, 80)', label: 'Very high (2.0-3.0 kPa)' },
    { color: 'rgb(255, 200, 100)', label: 'High (1.5-2.0 kPa)' },
    { color: 'rgb(200, 220, 150)', label: 'Moderate (1.0-1.5 kPa)' },
    { color: 'rgb(150, 200, 200)', label: 'Slight (0.5-1.0 kPa)' },
    { color: 'rgb(100, 150, 255)', label: 'Low stress (< 0.5 kPa)' },
  ],

  modis_lst_day: [
    { color: 'rgb(200, 50, 50)', label: 'Very hot (> 45°C)' },
    { color: 'rgb(255, 150, 80)', label: 'Hot (35-45°C)' },
    { color: 'rgb(255, 255, 100)', label: 'Warm (25-35°C)' },
    { color: 'rgb(150, 255, 150)', label: 'Mild (15-25°C)' },
    { color: 'rgb(100, 200, 255)', label: 'Cool (5-15°C)' },
    { color: 'rgb(150, 150, 255)', label: 'Cold (< 5°C)' },
  ],

  modis_lst_night: [
    { color: 'rgb(255, 180, 120)', label: 'Warm (> 25°C)' },
    { color: 'rgb(255, 220, 150)', label: 'Mild (20-25°C)' },
    { color: 'rgb(200, 220, 180)', label: 'Cool (15-20°C)' },
    { color: 'rgb(150, 200, 200)', label: 'Chilly (10-15°C)' },
    { color: 'rgb(120, 180, 255)', label: 'Cold (0-10°C)' },
    { color: 'rgb(150, 150, 255)', label: 'Freezing (< 0°C)' },
  ],
}
