import { useCallback, useEffect, useMemo, useState } from 'react'
import { CRS } from 'leaflet'
import { CircleMarker, MapContainer, TileLayer, useMapEvent } from 'react-leaflet'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const apiFetch = (path, options) => fetch(`${API_BASE}${path}`, options)
import { legends } from './utils/colorScales'

// Center on Morocco including Western Sahara
const mapCenter = [27.0, -8.5]

// Morocco bounding box - restricts map view to Morocco only
// [southwest corner, northeast corner]
const moroccoBounds = [
  [20.0, -17.5],  // Southwest (southern tip, western edge)
  [36.5, -0.5],   // Northeast (northern tip, eastern edge)
]

// Use Stamen Toner Lite - minimal base map for clean data visualization
const tileUrl = 'https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png'

// Available datasets with descriptions
const datasets = {
  modis_ndvi: {
    name: 'MODIS NDVI',
    description: 'Normalized Difference Vegetation Index (1km resolution)',
    unit: 'Index (unitless)',
    cadence: 'MONTHLY',
    agg: 'mean',
    startYear: 2000,
  },
  modis_evi: {
    name: 'MODIS EVI',
    description: 'Enhanced Vegetation Index (1km resolution)',
    unit: 'Index (unitless)',
    cadence: 'MONTHLY',
    agg: 'mean',
    startYear: 2000,
  },
  chirps_precip: {
    name: 'CHIRPS Precipitation',
    description: 'Daily precipitation (5km resolution)',
    unit: 'mm/day',
    cadence: 'DAILY',
    agg: 'sum',
    startYear: 1981,
  },
  era5_temp2m: {
    name: 'ERA5 Temperature',
    description: '2m air temperature monthly mean (11km resolution)',
    unit: '°C',
    cadence: 'MONTHLY',
    agg: 'mean',
    startYear: 1950,
  },
  era5_soil_moisture: {
    name: 'ERA5 Soil Moisture',
    description: 'Soil moisture layer 1 monthly mean (11km resolution)',
    unit: 'm³/m³',
    cadence: 'MONTHLY',
    agg: 'mean',
    startYear: 1950,
  },
  era5_vpd: {
    name: 'ERA5 VPD',
    description: 'Vapor Pressure Deficit monthly mean (11km resolution)',
    unit: 'kPa',
    cadence: 'MONTHLY',
    agg: 'mean',
    startYear: 1950,
  },
  modis_lst_day: {
    name: 'MODIS LST Day',
    description: '8-day composite land surface temperature (daytime, 1km)',
    unit: '°C (derived)',
    cadence: '8D',
    agg: 'mean',
    startYear: 2000,
  },
  modis_lst_night: {
    name: 'MODIS LST Night',
    description: '8-day composite land surface temperature (nighttime, 1km)',
    unit: '°C (derived)',
    cadence: '8D',
    agg: 'mean',
    startYear: 2000,
  },
}

// Generate months
const months = [
  { value: '01', label: 'January' },
  { value: '02', label: 'February' },
  { value: '03', label: 'March' },
  { value: '04', label: 'April' },
  { value: '05', label: 'May' },
  { value: '06', label: 'June' },
  { value: '07', label: 'July' },
  { value: '08', label: 'August' },
  { value: '09', label: 'September' },
  { value: '10', label: 'October' },
  { value: '11', label: 'November' },
  { value: '12', label: 'December' },
]

const now = new Date()
const currentYearToday = now.getFullYear()
const currentMonthIndex = now.getMonth()
const currentDayOfMonth = now.getDate()

const previousDate = new Date(currentYearToday, currentMonthIndex - 1, 1)
const previousYear = previousDate.getFullYear()
const previousMonthIndex = previousDate.getMonth()
const daysInPreviousMonth = new Date(previousYear, previousMonthIndex + 1, 0).getDate()
const defaultDayForPrevious = Math.min(currentDayOfMonth, daysInPreviousMonth)
const defaultMonthValue = String(previousMonthIndex + 1).padStart(2, '0')
const defaultDayValue = String(defaultDayForPrevious).padStart(2, '0')

function MapClickHandler({ onMapClick }) {
  useMapEvent('click', (event) => {
    onMapClick({ lat: event.latlng.lat, lng: event.latlng.lng })
  })
  return null
}

function App() {
  const [selectedDataset, setSelectedDataset] = useState('modis_ndvi')
  const [selectedYear, setSelectedYear] = useState(previousYear)
  const [selectedMonth, setSelectedMonth] = useState(defaultMonthValue)
  const [selectedDay, setSelectedDay] = useState(defaultDayValue)
  const [selectedAggregate, setSelectedAggregate] = useState('native')
  const [tileInfo, setTileInfo] = useState(null)
  const [tileError, setTileError] = useState(null)
  const [isTileLoading, setIsTileLoading] = useState(false)
  const [clickedLocation, setClickedLocation] = useState(null)
  const [pointInfo, setPointInfo] = useState(null)
  const [isSampling, setIsSampling] = useState(false)
  const [samplingError, setSamplingError] = useState(null)
  const [authStatus, setAuthStatus] = useState({ initialized: false, usingUploadedKey: false, customServiceAccount: null })
  const [authError, setAuthError] = useState(null)
  const [isUploadingKey, setIsUploadingKey] = useState(false)
  const [uploadedKeyName, setUploadedKeyName] = useState(null)
  const [downloadError, setDownloadError] = useState(null)
  const [latInput, setLatInput] = useState('')
  const [lngInput, setLngInput] = useState('')
  const [inputError, setInputError] = useState(null)
  const [cropOptions, setCropOptions] = useState([])
  const [isLoadingCrops, setIsLoadingCrops] = useState(false)
  const [cropError, setCropError] = useState(null)
  const [selectedCrop, setSelectedCrop] = useState('')
  const [phInput, setPhInput] = useState('7.0')
  const [moInput, setMoInput] = useState('1.2')
  const [pInput, setPInput] = useState('30')
  const [kInput, setKInput] = useState('400')
  const [expectedYieldInput, setExpectedYieldInput] = useState('40')
  const [fertimapResult, setFertimapResult] = useState(null)
  const [isFetchingRecommendation, setIsFetchingRecommendation] = useState(false)
  const [recommendationError, setRecommendationError] = useState(null)
  const [fertimapDefaults, setFertimapDefaults] = useState(null)
  const [fertimapYieldInfo, setFertimapYieldInfo] = useState(null)
  const [isLoadingFertimapDefaults, setIsLoadingFertimapDefaults] = useState(false)
  const [fertimapDefaultsError, setFertimapDefaultsError] = useState(null)
  const [pendingDefaultCrop, setPendingDefaultCrop] = useState(null)

  const currentDataset = datasets[selectedDataset]
  const cadence = currentDataset.cadence || 'MONTHLY'
  const datasetAggType = currentDataset.agg || 'mean'
  const datasetStartYear = currentDataset.startYear || 2000

  const aggregateOptions = useMemo(() => {
    const options = []
    const nativeLabel =
      cadence === 'DAILY'
        ? 'Native daily'
        : cadence === '8D'
          ? 'Native 8-day composite'
          : 'Native monthly'

    const labelForPeriod = (period) => {
      if (datasetAggType === 'sum') {
        return period === 'monthly' ? 'Monthly total' : 'Yearly total'
      }
      if (datasetAggType === 'median') {
        return period === 'monthly' ? 'Monthly median' : 'Yearly median'
      }
      return period === 'monthly' ? 'Monthly mean' : 'Yearly mean'
    }

    options.push({ value: 'native', label: nativeLabel })
    if (cadence !== 'MONTHLY') {
      options.push({ value: 'monthly', label: labelForPeriod('monthly') })
    }
    options.push({ value: 'yearly', label: labelForPeriod('yearly') })
    return options
  }, [cadence, datasetAggType])

  const aggregateOption = aggregateOptions.find((opt) => opt.value === selectedAggregate)
  const effectiveAggregate = aggregateOption ? selectedAggregate : 'native'
  const requiresDay = effectiveAggregate === 'native' && cadence !== 'MONTHLY'

  const availableYears = useMemo(() => {
    const start = Math.max(datasetStartYear, 1900)
    const yearsList = []
    for (let year = start; year <= currentYearToday; year += 1) {
      yearsList.push(year)
    }
    return yearsList
  }, [datasetStartYear])

  useEffect(() => {
    if (availableYears.length === 0) return
    if (!availableYears.includes(selectedYear)) {
      setSelectedYear(availableYears[availableYears.length - 1])
    }
  }, [availableYears, selectedYear])

  const availableMonths = useMemo(() => {
    if (selectedYear === currentYearToday) {
      return months.filter((month) => Number(month.value) <= currentMonthIndex + 1)
    }
    return months
  }, [selectedYear])

  useEffect(() => {
    if (availableMonths.length === 0) return
    if (!availableMonths.some((month) => month.value === selectedMonth)) {
      const fallback = availableMonths[availableMonths.length - 1]
      if (fallback) {
        setSelectedMonth(fallback.value)
      }
    }
  }, [availableMonths, selectedMonth])

  const handleMapClick = useCallback((latlng) => {
    setClickedLocation(latlng)
    setSamplingError(null)
  }, [])

  const formatValue = useCallback((raw) => {
    if (raw === null || raw === undefined) return 'No data'
    if (typeof raw === 'number') {
      if (Number.isNaN(raw) || !Number.isFinite(raw)) return 'No data'
      if (Math.abs(raw) >= 1000) return raw.toFixed(0)
      if (Math.abs(raw) >= 100) return raw.toFixed(1)
      return raw.toFixed(3)
    }
    return String(raw)
  }, [])

  const fetchAuthStatus = useCallback(async () => {
    try {
      const response = await apiFetch('/api/auth/status')
      if (!response.ok) return
      const data = await response.json()
      setAuthStatus({
        initialized: Boolean(data.initialized),
        usingUploadedKey: Boolean(data.usingUploadedKey),
        customServiceAccount: data.customServiceAccount || null,
      })
    } catch {
      // ignore fetch errors silently
    }
  }, [])

  const handleKeyUpload = useCallback(async (event) => {
    const file = event.target.files && event.target.files[0]
    if (!file) return
    setIsUploadingKey(true)
    setAuthError(null)
    try {
      const text = await file.text()
      const response = await apiFetch('/api/auth/service-account', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: text }),
      })
      if (!response.ok) {
        let message = response.statusText || 'Failed to upload key'
        try {
          const json = await response.json()
          if (json?.detail) {
            message = Array.isArray(json.detail) ? json.detail.join(', ') : json.detail
          }
        } catch {
          // ignore JSON parse error
        }
        throw new Error(message)
      }
      setUploadedKeyName(file.name)
      await fetchAuthStatus()
    } catch (error) {
      setAuthError(error.message || 'Failed to upload key')
    } finally {
      setIsUploadingKey(false)
      event.target.value = ''
    }
  }, [fetchAuthStatus])

  const handleDownload = useCallback(async () => {
    const monthNumber = Number(selectedMonth)
    const dayNumber = requiresDay ? Number(selectedDay) : null
    setDownloadError(null)
    try {
      const params = new URLSearchParams({
        dataset: selectedDataset,
        year: String(selectedYear),
        month: String(monthNumber),
        aggregate: effectiveAggregate,
      })
      if (requiresDay && dayNumber) {
        params.append('day', String(dayNumber))
      }
      const response = await apiFetch(`/api/download?${params.toString()}`)
      if (!response.ok) {
        let message = response.statusText || 'Failed to prepare download'
        try {
          const json = await response.json()
          if (json?.detail) {
            message = Array.isArray(json.detail) ? json.detail.join(', ') : json.detail
          }
        } catch {
          // ignore JSON parse errors
        }
        throw new Error(message)
      }
      const data = await response.json()
      const link = document.createElement('a')
      link.href = data.url
      link.download = data.fileName || 'gee_tile.tif'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (error) {
      setDownloadError(error.message || 'Failed to prepare download')
    }
  }, [selectedDataset, selectedYear, selectedMonth, selectedDay, requiresDay, effectiveAggregate])

  const handleCoordinateSubmit = useCallback((event) => {
    event.preventDefault()
    setInputError(null)
    const parsedLat = Number(latInput)
    const parsedLng = Number(lngInput)
    if (Number.isNaN(parsedLat) || Number.isNaN(parsedLng)) {
      setInputError('Latitude and longitude must be numeric values.')
      return
    }
    if (parsedLat < -90 || parsedLat > 90 || parsedLng < -180 || parsedLng > 180) {
      setInputError('Latitude must be between -90 and 90, longitude between -180 and 180.')
      return
    }
    setClickedLocation({ lat: parsedLat, lng: parsedLng })
    setSamplingError(null)
  }, [latInput, lngInput])

  const handleRecommendationSubmit = useCallback(async (event) => {
    event.preventDefault()
    setRecommendationError(null)

    const parsedLat = Number(latInput)
    const parsedLng = Number(lngInput)
    if (Number.isNaN(parsedLat) || Number.isNaN(parsedLng)) {
      setRecommendationError('Enter a valid latitude and longitude before requesting recommendations.')
      return
    }
    if (parsedLat < -90 || parsedLat > 90 || parsedLng < -180 || parsedLng > 180) {
      setRecommendationError('Latitude must be between -90 and 90, longitude between -180 and 180.')
      return
    }
    if (!selectedCrop) {
      setRecommendationError('Select a crop to request a recommendation.')
      return
    }

    const phValue = Number(phInput)
    const moValue = Number(moInput)
    const pValue = Number(pInput)
    const kValue = Number(kInput)
    const expectedYieldValue = Number(expectedYieldInput)

    const numericChecks = [
      ['pH', phValue],
      ['Organic matter', moValue],
      ['Phosphorus', pValue],
      ['Potassium', kValue],
      ['Expected yield', expectedYieldValue],
    ]

    for (const [label, value] of numericChecks) {
      if (Number.isNaN(value)) {
        setRecommendationError(`${label} must be a numeric value.`)
        return
      }
    }

    if (expectedYieldValue <= 0) {
      setRecommendationError('Expected yield must be greater than zero.')
      return
    }

    setIsFetchingRecommendation(true)

    try {
      const response = await apiFetch('/api/fertimap/recommendation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: parsedLat,
          lng: parsedLng,
          ph: phValue,
          mo: moValue,
          p: pValue,
          k: kValue,
          crop_name: selectedCrop,
          expected_yield: expectedYieldValue,
        }),
      })

      if (!response.ok) {
        let message = response.statusText || 'Failed to fetch recommendation'
        try {
          const json = await response.json()
          if (json?.detail) {
            message = Array.isArray(json.detail) ? json.detail.join(', ') : json.detail
          }
        } catch {
          // ignore JSON parse errors
        }
        throw new Error(message)
      }

      const data = await response.json()
      setFertimapResult(data)
    } catch (error) {
      setFertimapResult(null)
      setRecommendationError(error.message || 'Failed to fetch recommendation')
    } finally {
      setIsFetchingRecommendation(false)
    }
  }, [
    latInput,
    lngInput,
    phInput,
    moInput,
    pInput,
    kInput,
    expectedYieldInput,
    selectedCrop,
  ])

  const daysInMonth = useMemo(() => {
    const monthNumber = Number(selectedMonth)
    return new Date(selectedYear, monthNumber, 0).getDate()
  }, [selectedYear, selectedMonth])

  const maxSelectableDay = useMemo(() => {
    const monthNumber = Number(selectedMonth)
    let maxDay = daysInMonth
    if (selectedYear === currentYearToday && monthNumber === currentMonthIndex + 1) {
      maxDay = Math.min(maxDay, currentDayOfMonth)
    }
    return maxDay
  }, [daysInMonth, selectedYear, selectedMonth])

  const dayOptions = useMemo(() => {
    if (!requiresDay) return []
    if (cadence === 'DAILY') {
      return Array.from({ length: maxSelectableDay }, (_, i) => String(i + 1).padStart(2, '0'))
    }
    if (cadence === '8D') {
      const options = []
      for (let day = 1; day <= maxSelectableDay; day += 8) {
        options.push(String(day).padStart(2, '0'))
      }
      return options
    }
    return []
  }, [requiresDay, cadence, maxSelectableDay])
  
  const monthMeta = months.find((m) => m.value === selectedMonth)
  const monthLabel = monthMeta ? monthMeta.label : selectedMonth
  const dateLabel =
    effectiveAggregate === 'yearly'
      ? `Year ${selectedYear}`
      : requiresDay
        ? `${monthLabel} ${Number(selectedDay)}, ${selectedYear}`
        : `${monthLabel} ${selectedYear}`
  const nativeCadenceLabel =
    cadence === '8D' ? '8-day composite' : cadence === 'DAILY' ? 'Daily' : 'Monthly'
  const aggregationLabel = aggregateOption?.label || aggregateOptions[0]?.label || 'Native'
  const aggregateSelectValue = aggregateOption ? selectedAggregate : 'native'

  // Create a unique key to force component remount when parameters change
  const timeKey =
    effectiveAggregate === 'yearly'
      ? 'year'
      : requiresDay
        ? `${selectedMonth}-${selectedDay}`
        : selectedMonth
  const layerKey = `${selectedDataset}-${effectiveAggregate}-${selectedYear}-${timeKey}`
  
  // Get the legend items for the current dataset
  const legendItems = useMemo(
    () => (tileInfo?.legend?.length ? tileInfo.legend : legends[selectedDataset] || []),
    [tileInfo, selectedDataset]
  )

  useEffect(() => {
    if (!aggregateOption && selectedAggregate !== 'native') {
      setSelectedAggregate('native')
    }
  }, [aggregateOption, selectedAggregate])

  useEffect(() => {
    if (clickedLocation) {
      setLatInput(clickedLocation.lat.toFixed(5))
      setLngInput(clickedLocation.lng.toFixed(5))
    }
  }, [clickedLocation])

  useEffect(() => {
    fetchAuthStatus()
  }, [fetchAuthStatus])

  useEffect(() => {
    let cancelled = false
    const loadCrops = async () => {
      setIsLoadingCrops(true)
      setCropError(null)
      try {
        const response = await apiFetch('/api/fertimap/crops')
        if (!response.ok) {
          let message = response.statusText || 'Failed to load crop list'
          try {
            const json = await response.json()
            if (json?.detail) {
              message = Array.isArray(json.detail) ? json.detail.join(', ') : json.detail
            }
          } catch {
            // ignore JSON parse errors
          }
          throw new Error(message)
        }
        const data = await response.json()
        if (cancelled) return
        const crops = Array.isArray(data?.crops) ? data.crops : []
        setCropOptions(crops)
      } catch (error) {
        if (!cancelled) {
          setCropError(error.message || 'Failed to load crop list')
        }
      } finally {
        if (!cancelled) {
          setIsLoadingCrops(false)
        }
      }
    }

    loadCrops()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const availableNames = new Set(cropOptions.map((crop) => crop.name))
    if (cropOptions.length === 0) {
      if (selectedCrop !== '') {
        setSelectedCrop('')
      }
      return
    }

    let nextCrop = selectedCrop
    let appliedPending = false

    if (pendingDefaultCrop && availableNames.has(pendingDefaultCrop)) {
      nextCrop = pendingDefaultCrop
      appliedPending = true
    } else if (!availableNames.has(nextCrop)) {
      nextCrop = cropOptions[0]?.name || ''
    }

    if (nextCrop !== selectedCrop) {
      setSelectedCrop(nextCrop)
    }

    if (appliedPending) {
      setPendingDefaultCrop(null)
    }
  }, [cropOptions, pendingDefaultCrop, selectedCrop])

  useEffect(() => {
    if (!clickedLocation) {
      return
    }
    let cancelled = false
    const loadDefaults = async () => {
      setIsLoadingFertimapDefaults(true)
      setFertimapDefaultsError(null)
      try {
        const params = new URLSearchParams({
          lat: clickedLocation.lat.toFixed(5),
          lng: clickedLocation.lng.toFixed(5),
        })
        const response = await apiFetch(`/api/fertimap/defaults?${params.toString()}`)
        if (!response.ok) {
          let message = response.statusText || 'Failed to load Fertimap defaults'
          try {
            const json = await response.json()
            if (json?.detail) {
              message = Array.isArray(json.detail) ? json.detail.join(', ') : json.detail
            }
          } catch {
            // ignore JSON parse errors
          }
          throw new Error(message)
        }
        const data = await response.json()
        if (cancelled) return
        setFertimapDefaults(data?.defaults || null)
      } catch (error) {
        if (!cancelled) {
          setFertimapDefaults(null)
          setFertimapDefaultsError(error.message || 'Failed to load Fertimap defaults')
        }
      } finally {
        if (!cancelled) {
          setIsLoadingFertimapDefaults(false)
        }
      }
    }

    loadDefaults()
    return () => {
      cancelled = true
    }
  }, [clickedLocation])

  useEffect(() => {
    if (!fertimapDefaults) {
      setFertimapYieldInfo(null)
      return
    }

    const formatValueForInput = (value) => {
      if (value === null || value === undefined) return null
      if (Number.isNaN(Number(value))) return null
      const numeric = Number(value)
      const rounded = Math.round(numeric * 1000) / 1000
      if (Number.isNaN(rounded)) return null
      if (Math.abs(rounded - Math.round(rounded)) < 1e-6) {
        return String(Math.round(rounded))
      }
      return String(rounded)
    }

    const applyInputValue = (setter, value) => {
      const formatted = formatValueForInput(value)
      if (formatted == null) return
      setter((prev) => (prev === formatted ? prev : formatted))
    }

    applyInputValue(setPhInput, fertimapDefaults.ph)
    applyInputValue(setMoInput, fertimapDefaults.mo)
    applyInputValue(setPInput, fertimapDefaults.p)
    applyInputValue(setKInput, fertimapDefaults.k)

    const yieldMin = fertimapDefaults.yield_min ?? null
    const yieldMax = fertimapDefaults.yield_max ?? null
    const yieldStep = fertimapDefaults.yield_step ?? null
    const yieldUnit = fertimapDefaults.yield_unit ?? ''
    let expectedYield = fertimapDefaults.expected_yield ?? null

    if (expectedYield != null) {
      if (yieldMin != null) {
        expectedYield = Math.max(expectedYield, yieldMin)
      }
      if (yieldMax != null) {
        expectedYield = Math.min(expectedYield, yieldMax)
      }
      applyInputValue(setExpectedYieldInput, expectedYield)
    }

    if (yieldMin != null || yieldMax != null || yieldStep != null || yieldUnit) {
      setFertimapYieldInfo({
        min: yieldMin,
        max: yieldMax,
        step: yieldStep,
        unit: yieldUnit,
      })
    } else {
      setFertimapYieldInfo(null)
    }

    if (fertimapDefaults.crop_name) {
      setPendingDefaultCrop(fertimapDefaults.crop_name)
    }
  }, [fertimapDefaults])

  useEffect(() => {
    if (!requiresDay) {
      setSelectedDay('01')
      return
    }
    if (dayOptions.length > 0 && !dayOptions.includes(selectedDay)) {
      setSelectedDay(dayOptions[0])
    }
  }, [requiresDay, dayOptions, selectedDay])

  useEffect(() => {
    const controller = new AbortController()

    const fetchTileMetadata = async () => {
      const monthNumber = Number(selectedMonth)
      const dayNumber = requiresDay ? Number(selectedDay) : null
      setIsTileLoading(true)
      setTileError(null)
      try {
        const params = new URLSearchParams({
          dataset: selectedDataset,
          year: String(selectedYear),
          month: String(monthNumber),
        })
        params.append('aggregate', effectiveAggregate)
        if (requiresDay && dayNumber) {
          params.append('day', String(dayNumber))
        }
        const response = await apiFetch(`/api/tiles?${params.toString()}`, {
          signal: controller.signal,
        })

        if (!response.ok) {
          let message = response.statusText || 'Failed to load tile metadata'
          try {
            const json = await response.json()
            if (json?.detail) {
              message = Array.isArray(json.detail) ? json.detail.join(', ') : json.detail
            }
          } catch {
            // ignore JSON parse errors
          }
          throw new Error(message)
        }

        const data = await response.json()
        setTileInfo(data)
      } catch (error) {
        if (error.name === 'AbortError') return
        setTileError(error.message || 'Failed to load tile metadata')
      } finally {
        if (!controller.signal.aborted) {
          setIsTileLoading(false)
        }
      }
    }

    fetchTileMetadata()

    return () => {
      controller.abort()
    }
  }, [selectedDataset, selectedYear, selectedMonth, requiresDay, selectedDay, effectiveAggregate])

  useEffect(() => {
    if (!clickedLocation) return

    const controller = new AbortController()
    const fetchSample = async () => {
      const monthNumber = Number(selectedMonth)
      const dayNumber = requiresDay ? Number(selectedDay) : null
      setIsSampling(true)
      setSamplingError(null)
      try {
        const params = new URLSearchParams({
          dataset: selectedDataset,
          year: String(selectedYear),
          month: String(monthNumber),
          aggregate: effectiveAggregate,
          lat: clickedLocation.lat.toFixed(6),
          lng: clickedLocation.lng.toFixed(6),
        })
        if (requiresDay && dayNumber) {
          params.append('day', String(dayNumber))
        }

        const response = await apiFetch(`/api/sample?${params.toString()}`, {
          signal: controller.signal,
        })

        if (!response.ok) {
          let message = response.statusText || 'Failed to sample pixel'
          try {
            const json = await response.json()
            if (json?.detail) {
              message = Array.isArray(json.detail) ? json.detail.join(', ') : json.detail
            }
          } catch {
            // ignore JSON parse errors
          }
          throw new Error(message)
        }

        const data = await response.json()
        setPointInfo(data)
      } catch (error) {
        if (error.name === 'AbortError') return
        setPointInfo(null)
        setSamplingError(error.message || 'Failed to sample pixel')
      } finally {
        if (!controller.signal.aborted) {
          setIsSampling(false)
        }
      }
    }

    fetchSample()

    return () => {
      controller.abort()
    }
  }, [
    clickedLocation,
    selectedDataset,
    selectedYear,
    selectedMonth,
    selectedDay,
    effectiveAggregate,
    requiresDay,
  ])

  const datasetTileKey = useMemo(
    () => (tileInfo?.token ? `${layerKey}-${tileInfo.token}` : `${layerKey}-loading`),
    [layerKey, tileInfo]
  )

  const fertimapLocation = fertimapResult?.location ?? null
  const fertimapRecommendation = fertimapResult?.recommendation ?? null

  const fertimapProductGroups = useMemo(() => {
    if (!fertimapRecommendation?.recommendations) {
      return []
    }

    const groups = [
      {
        key: 'regional',
        title: 'Regional formula',
        helper: 'Recommended baseline for the province.',
      },
      {
        key: 'selected_yield',
        title: 'Selected yield',
        helper: 'Optimised for your target yield.',
      },
      {
        key: 'generic',
        title: 'Generic formulas',
        helper: 'Standard fertiliser mixes you can source easily.',
      },
    ]

    return groups
      .map((group) => ({
        ...group,
        items: fertimapRecommendation.recommendations[group.key] || [],
      }))
      .filter((group) => group.items.length > 0)
  }, [fertimapRecommendation])

  const fertimapYieldRangeLabel = useMemo(() => {
    if (!fertimapYieldInfo) return null
    const formatNumber = (value) => {
      if (value === null || value === undefined) return null
      const numeric = Number(value)
      if (Number.isNaN(numeric)) return null
      const rounded = Math.round(numeric * 100) / 100
      if (Math.abs(rounded - Math.round(rounded)) < 1e-6) {
        return String(Math.round(rounded))
      }
      return rounded.toString()
    }

    const unit = fertimapYieldInfo.unit ? ` ${fertimapYieldInfo.unit}` : ''
    const minLabel = formatNumber(fertimapYieldInfo.min)
    const maxLabel = formatNumber(fertimapYieldInfo.max)

    if (minLabel && maxLabel) {
      return `${minLabel} – ${maxLabel}${unit}`
    }
    if (minLabel) {
      return `≥ ${minLabel}${unit}`
    }
    if (maxLabel) {
      return `≤ ${maxLabel}${unit}`
    }
    return unit.trim() ? unit.trim() : null
  }, [fertimapYieldInfo])

  const clickedLat = clickedLocation?.lat ?? null
  const clickedLng = clickedLocation?.lng ?? null
  const pointRangeLabel =
    pointInfo?.startDate && pointInfo?.endDate
      ? pointInfo.startDate === pointInfo.endDate
        ? pointInfo.startDate
        : `${pointInfo.startDate} → ${pointInfo.endDate}`
      : null
  const sampleAggregateLabel = pointInfo?.aggregate
    ? aggregateOptions.find((option) => option.value === pointInfo.aggregate)?.label || pointInfo.aggregate
    : aggregationLabel

  const globalErrors = [
    !authStatus.usingUploadedKey
      ? { key: 'auth-required', message: 'Upload a Google Earth Engine service account key to enable the map.' }
      : null,
    authError ? { key: 'auth', message: authError } : null,
    tileError ? { key: 'tiles', message: tileError } : null,
    samplingError ? { key: 'sample', message: samplingError } : null,
    downloadError ? { key: 'download', message: downloadError } : null,
    inputError ? { key: 'input', message: inputError } : null,
    fertimapDefaultsError ? { key: 'fertimap-defaults', message: fertimapDefaultsError } : null,
  ].filter(Boolean)

  return (
    <div className="app-shell">
      <header className="headline">
        <div className="headline-text">
          <p className="eyebrow">Morocco Geospatial Data Viewer</p>
          <h1>Fertilizer Recommendation System on Moroccan Soil</h1>
          <p className="lede">
            Interactive visualization of satellite and climate data for Morocco.
            Select a dataset, year, and month to explore temporal patterns across
            the region.
          </p>
        </div>
        <div className="hero-brand">
          <img src="/data/logo.svg" alt="Project logo" className="brand-mark" />
          {/* <img src="/data/cc-logo.png" alt="Partner logo" className="brand-mark" /> */}
        </div>
      </header>

      <section className="info-grid">
        <article className="info-card">
          <span className="info-label">Authentication</span>
          <h3 className="info-title">
            {authStatus.usingUploadedKey
              ? 'Custom key active'
              : 'Key required'}
          </h3>
          <p className="info-body">
            {authStatus.usingUploadedKey
              ? 'Tiles are streaming with your uploaded service account key.'
              : 'Upload a Google Earth Engine service account key JSON to authenticate with your credentials.'}
          </p>
          <div className="upload-row">
            <label className="upload-button">
              <input
                type="file"
                accept="application/json"
                onChange={handleKeyUpload}
                disabled={isUploadingKey}
              />
              {isUploadingKey ? 'Uploading…' : 'Upload key JSON'}
            </label>
          </div>
          <div className="info-chips">
            {authStatus.customServiceAccount && (
              <span className="chip">{authStatus.customServiceAccount}</span>
            )}
            {uploadedKeyName && authStatus.usingUploadedKey && (
              <span className="chip">Last upload: {uploadedKeyName}</span>
            )}
          </div>
        </article>

        <article className="info-card">
          <span className="info-label">Dataset</span>
          <h2 className="info-title">{currentDataset.name}</h2>
          <p className="info-body">{currentDataset.description}</p>
          <div className="info-chips">
            {currentDataset.unit && <span className="chip">Unit: {currentDataset.unit}</span>}
            <span className="chip">Native cadence: {nativeCadenceLabel}</span>
            {datasetStartYear && (
              <span className="chip">Coverage: {datasetStartYear}–{currentYearToday}</span>
            )}
          </div>
        </article>
        <article className="info-card">
          <span className="info-label">Selection</span>
          <h3 className="info-title">{dateLabel}</h3>
          <ul className="info-list">
            <li><strong>Aggregation:</strong> {aggregationLabel}</li>
            {tileInfo?.startDate && tileInfo?.endDate && (
              <li>
                <strong>Period:</strong>{' '}
                {tileInfo.startDate === tileInfo.endDate
                  ? tileInfo.startDate
                  : `${tileInfo.startDate} → ${tileInfo.endDate}`}
              </li>
            )}
            <li><strong>Resolution:</strong> {tileInfo?.scaleMeters ? `${Math.round(tileInfo.scaleMeters)} m` : 'N/A'}</li>
          </ul>
        </article>
        <article className="info-card">
          <span className="info-label">Pixel Sample</span>
          {clickedLocation && pointInfo ? (
            <ul className="info-list compact">
              <li><strong>Lat/Lng:</strong> {clickedLat?.toFixed(5)}, {clickedLng?.toFixed(5)}</li>
              <li><strong>Value:</strong> {formatValue(pointInfo.value)}{typeof pointInfo.value === 'number' && pointInfo.unit ? ` ${pointInfo.unit}` : ''}</li>
              <li><strong>Aggregation:</strong> {sampleAggregateLabel}</li>
              {pointRangeLabel && <li><strong>Period:</strong> {pointRangeLabel}</li>}
              <li><strong>Elevation:</strong> {pointInfo?.elevationM != null ? `${Math.round(pointInfo.elevationM)} m` : 'N/A'}</li>
              <li><strong>Land cover:</strong> {pointInfo?.landCoverLabel || 'N/A'}{pointInfo?.landCoverCode != null ? ` (${pointInfo.landCoverCode})` : ''}</li>
              <li><strong>Province:</strong> {pointInfo?.admin1Name || 'N/A'}</li>
              <li><strong>District:</strong> {pointInfo?.admin2Name || 'N/A'}</li>
            </ul>
          ) : (
            <p className="info-body subtle">Click the map or enter coordinates to inspect a pixel.</p>
          )}
        </article>
      </section>

      {globalErrors.length > 0 && (
        <div className="alert-stack">
          {globalErrors.map((error, index) => (
            <div key={`${error.key}-${index}`} className="alert-card">
              {error.message}
            </div>
          ))}
        </div>
      )}

      <section className="controls-bar">
        <div className="controls-row">
          <label htmlFor="dataset-select">Dataset
            <select
              id="dataset-select"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="control-select"
            >
              {Object.entries(datasets).map(([key, ds]) => (
                <option key={key} value={key}>
                  {ds.name}
                </option>
              ))}
            </select>
          </label>

          <label htmlFor="aggregate-select">Aggregation
            <select
              id="aggregate-select"
              value={aggregateSelectValue}
              onChange={(e) => setSelectedAggregate(e.target.value)}
              className="control-select"
            >
              {aggregateOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label htmlFor="year-select">Year
            <select
              id="year-select"
              value={selectedYear}
              onChange={(e) => setSelectedYear(Number(e.target.value))}
              className="control-select"
            >
              {availableYears.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
          </label>

          <label htmlFor="month-select">Month
            <select
              id="month-select"
              value={selectedMonth}
              onChange={(e) => setSelectedMonth(e.target.value)}
              className="control-select"
              disabled={effectiveAggregate === 'yearly'}
            >
              {availableMonths.map((month) => (
                <option key={month.value} value={month.value}>
                  {month.label}
                </option>
              ))}
            </select>
          </label>

          {requiresDay && (
            <label htmlFor="day-select">{cadence === '8D' ? 'Start day' : 'Day'}
              <select
                id="day-select"
                value={selectedDay}
                onChange={(e) => setSelectedDay(e.target.value)}
                className="control-select"
              >
                {dayOptions.map((day) => (
                  <option key={day} value={day}>
                    {Number(day)}
                  </option>
                ))}
              </select>
            </label>
          )}

          <button
            type="button"
            className="ghost-button"
            onClick={handleDownload}
            disabled={!authStatus.usingUploadedKey || isTileLoading}
          >
            Download GeoTIFF
          </button>
        </div>
      </section>

      <section className="map-layout">
        <div className="map-column">
          <div className="map-wrapper">
            {/* Use key to force MapContainer to remount when layer changes */}
            <MapContainer
              key={layerKey}
              center={mapCenter}
              zoom={5}
              minZoom={4}
              maxZoom={11}
              maxBounds={moroccoBounds}
              maxBoundsViscosity={1.0}
              scrollWheelZoom
              className="morocco-map"
            >
              <TileLayer
                url={tileUrl}
                attribution='&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                eventHandlers={{
                  tileerror: (e) => {
                    console.warn('Leaflet tile error', e)
                  },
                }}
              />

              {tileInfo?.tileUrl && (
                <TileLayer
                  key={datasetTileKey}
                  url={tileInfo.tileUrl}
                  opacity={0.85}
                  attribution={tileInfo.attribution}
                />
              )}
              <MapClickHandler onMapClick={handleMapClick} />
              {clickedLocation && (
                <CircleMarker
                  center={[clickedLocation.lat, clickedLocation.lng]}
                  radius={6}
                  pathOptions={{ color: '#052c23ff', weight: 2, fillOpacity: 0.4 }}
                />
              )}
            </MapContainer>

            <div className="map-status-overlay">
              {isTileLoading && <span className="status loading">Loading layer…</span>}
              {isSampling && <span className="status loading">Sampling pixel…</span>}
            </div>
          </div>
        </div>

        <aside className="controls-panel">
          <div className="control-card">
            <h3>Legend</h3>
            <div className="legend-grid">
              {legendItems.map((item, index) => (
                <div key={index} className="legend-item">
                  <span className="legend-color" style={{ backgroundColor: item.color }}></span>
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="control-card">
            <h3>Coordinate Lookup</h3>
            <p className="layer-meta">Enter latitude and longitude to sample any location.</p>
            <form className="coordinate-form" onSubmit={handleCoordinateSubmit}>
              <label>
                Latitude
                <input
                  type="text"
                  value={latInput}
                  onChange={(e) => setLatInput(e.target.value)}
                  className="text-input"
                  placeholder="e.g. 30.5"
                />
              </label>
              <label>
                Longitude
                <input
                  type="text"
                  value={lngInput}
                  onChange={(e) => setLngInput(e.target.value)}
                  className="text-input"
                  placeholder="e.g. -7.4"
                />
              </label>
              <button type="submit" className="primary-button">Sample location</button>
            </form>
          </div>
        </aside>
      </section>

      <section className="fertimap-section">
        <div className="section-header">
          <h2>Fertimap Nutrient Planner</h2>
          <p>
            Combine the map coordinates with your soil analysis to retrieve fertiliser guidance published by
            the Moroccan Fertimap service.
          </p>
        </div>

        <div className="fertimap-grid">
          <form className="fertimap-form" onSubmit={handleRecommendationSubmit}>
            <h3 className="fertimap-form-title">Inputs</h3>
            <p className="form-hint">
              Coordinates stay in sync with the pixel sample controls above. Adjust them here or by clicking the map.
            </p>
            {isLoadingFertimapDefaults ? (
              <p className="subtle-text">Loading soil defaults for this location…</p>
            ) : fertimapDefaults ? (
              <p className="subtle-text">Inputs auto-filled from Fertimap soil survey data.</p>
            ) : null}

            <div className="form-row">
              <label>
                Latitude
                <input
                  type="text"
                  value={latInput}
                  onChange={(e) => setLatInput(e.target.value)}
                  className="text-input"
                  placeholder="e.g. 31.9631"
                />
              </label>
              <label>
                Longitude
                <input
                  type="text"
                  value={lngInput}
                  onChange={(e) => setLngInput(e.target.value)}
                  className="text-input"
                  placeholder="e.g. -8.6106"
                />
              </label>
            </div>

            <label>
              Crop
              <select
                value={selectedCrop}
                onChange={(e) => setSelectedCrop(e.target.value)}
                className="control-select"
                disabled={isLoadingCrops || cropOptions.length === 0}
              >
                {cropOptions.map((crop) => (
                  <option key={crop.id} value={crop.name}>
                    {crop.name}
                  </option>
                ))}
              </select>
            </label>
            {isLoadingCrops && <p className="form-hint subtle">Loading crop list…</p>}

            <div className="form-grid">
              <label>
                Soil pH
                <input
                  type="number"
                  step="0.1"
                  value={phInput}
                  onChange={(e) => setPhInput(e.target.value)}
                  className="text-input"
                />
              </label>
              <label>
                Organic matter (%)
                <input
                  type="number"
                  step="0.1"
                  value={moInput}
                  onChange={(e) => setMoInput(e.target.value)}
                  className="text-input"
                />
              </label>
              <label>
                Phosphorus (P₂O₅ mg/kg)
                <input
                  type="number"
                  step="0.1"
                  value={pInput}
                  onChange={(e) => setPInput(e.target.value)}
                  className="text-input"
                />
              </label>
              <label>
                Potassium (K₂O mg/kg)
                <input
                  type="number"
                  step="0.1"
                  value={kInput}
                  onChange={(e) => setKInput(e.target.value)}
                  className="text-input"
                />
              </label>
              <label>
                Expected yield (qx/ha)
                <input
                  type="number"
                  step="1"
                  value={expectedYieldInput}
                  onChange={(e) => setExpectedYieldInput(e.target.value)}
                  className="text-input"
                />
              </label>
            </div>
            {fertimapYieldRangeLabel && (
              <p className="subtle-text">Suggested yield range: {fertimapYieldRangeLabel}</p>
            )}

            <button
              type="submit"
              className="primary-button"
              disabled={!selectedCrop || isFetchingRecommendation}
            >
              {isFetchingRecommendation ? 'Requesting…' : 'Get recommendation'}
            </button>

            {cropError && <p className="form-error">{cropError}</p>}
            {recommendationError && <p className="form-error">{recommendationError}</p>}
          </form>

          <div className="fertimap-output">
            {isFetchingRecommendation && !fertimapRecommendation && (
              <p className="form-hint subtle">Fetching recommendation…</p>
            )}

            {fertimapRecommendation ? (
              <div className="fertimap-results">
                <div className="fertimap-metrics">
                  <div className="metric-card">
                    <span className="metric-label">Nitrogen (N)</span>
                    <span className="metric-value">
                      {fertimapRecommendation.N != null
                        ? `${formatValue(fertimapRecommendation.N)} kg/ha`
                        : 'No data'}
                    </span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Phosphorus (P)</span>
                    <span className="metric-value">
                      {fertimapRecommendation.P != null
                        ? `${formatValue(fertimapRecommendation.P)} kg/ha`
                        : 'No data'}
                    </span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Potassium (K)</span>
                    <span className="metric-value">
                      {fertimapRecommendation.K != null
                        ? `${formatValue(fertimapRecommendation.K)} kg/ha`
                        : 'No data'}
                    </span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Estimated cost</span>
                    <span className="metric-value">
                      {fertimapRecommendation.cost != null
                        ? `${formatValue(fertimapRecommendation.cost)} dh/ha`
                        : 'No data'}
                    </span>
                  </div>
                </div>

                {fertimapLocation && (
                  <div className="fertimap-location">
                    <h3>Location context</h3>
                    <ul>
                      <li><strong>Region:</strong> {fertimapLocation.region || 'N/A'}</li>
                      <li><strong>Province:</strong> {fertimapLocation.province || 'N/A'}</li>
                      <li><strong>Commune:</strong> {fertimapLocation.commune || 'N/A'}</li>
                      <li><strong>Soil type:</strong> {fertimapLocation.soil_type || 'N/A'}</li>
                      <li><strong>Texture:</strong> {fertimapLocation.texture || 'N/A'}</li>
                    </ul>
                  </div>
                )}

                {fertimapProductGroups.length > 0 ? (
                  <div className="fertimap-products">
                    {fertimapProductGroups.map((group) => (
                      <div key={group.key} className="product-group">
                        <h4>{group.title}</h4>
                        <p className="form-hint">{group.helper}</p>
                        <ul>
                          {group.items.map((item, index) => (
                            <li key={`${group.key}-${index}`}>
                              <span className="product-quantity">{formatValue(item.quantity)} qx/ha</span>
                              <span className="product-name">{item.name}</span>
                              {item.type && <span className="product-tag">{item.type}</span>}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="form-hint subtle">
                    No specific fertiliser products were returned for this configuration. Try another crop or adjust the inputs.
                  </p>
                )}
              </div>
            ) : (
              !isFetchingRecommendation && (
                <p className="form-hint subtle">
                  Select a location, provide soil analysis results, then request a recommendation to see targeted nutrient needs.
                </p>
              )
            )}
          </div>
        </div>
      </section>

    </div>
  )
}

export default App
