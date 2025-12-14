import parseGeoraster from 'georaster'
import GeoRasterLayer from 'georaster-layer-for-leaflet'
import { useEffect, useRef } from 'react'
import { useMap } from 'react-leaflet'

/**
 * GeoTiffLayer component for react-leaflet
 * Loads a GeoTIFF file and displays it on the map with a color scale
 * Simplified version - parent MapContainer remounts with each change
 * 
 * @param {string} url - Path to the GeoTIFF file
 * @param {number} opacity - Layer opacity (0-1)
 * @param {function} pixelValuesToColorFn - Function to convert pixel values to colors
 * @param {number} resolution - Render resolution
 */
function GeoTiffLayer({ url, opacity = 0.85, pixelValuesToColorFn, resolution = 256 }) {
  const map = useMap()
  const layerRef = useRef(null)

  useEffect(() => {
    if (!url || !map) return

    let isMounted = true

    const loadGeoTiff = async () => {
      try {
        console.log(`Loading GeoTIFF from: ${url}`)
        
        // Fetch the GeoTIFF file
        const response = await fetch(url)
        if (!response.ok) {
          throw new Error(`Failed to fetch GeoTIFF: ${response.statusText}`)
        }
        
        const arrayBuffer = await response.arrayBuffer()
        
        // Parse the georaster
        const georaster = await parseGeoraster(arrayBuffer)
        
        console.log('GeoTIFF metadata:', {
          width: georaster.width,
          height: georaster.height,
          xmin: georaster.xmin,
          xmax: georaster.xmax,
          ymin: georaster.ymin,
          ymax: georaster.ymax,
        })

        if (!isMounted) return

        // Create the georaster layer
        const layer = new GeoRasterLayer({
          georaster,
          opacity,
          pixelValuesToColorFn,
          resolution,
        })

        layer.addTo(map)
        layerRef.current = layer
        console.log('✓ GeoTIFF layer added to map')
      } catch (error) {
        console.error('Error loading GeoTIFF:', error)
      }
    }

    loadGeoTiff()

    // Cleanup on unmount (MapContainer will remount for each new layer)
    return () => {
      isMounted = false
      if (layerRef.current && map.hasLayer(layerRef.current)) {
        map.removeLayer(layerRef.current)
      }
    }
  }, [url, map, opacity, pixelValuesToColorFn, resolution])

  return null
}

export default GeoTiffLayer
