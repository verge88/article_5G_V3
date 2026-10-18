package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
)

type Control struct {
	EventID      string  `json:"event_id"`
	Mode         string  `json:"mode"`
	DelayMS      int     `json:"delay_ms"`
	JitterMS     int     `json:"jitter_ms"`
	ErrorEveryN  int     `json:"error_every_n"`
	ExcludeToken string  `json:"exclude_token"`
	FaultRate    float64 `json:"fault_rate"`
}

type EventStats struct {
	EventID        string    `json:"event_id"`
	Requests       int       `json:"requests"`
	Business       int       `json:"business_requests"`
	Faulted        int       `json:"faulted_requests"`
	InjectedErrors int       `json:"injected_errors"`
	DurationsMS    []float64 `json:"durations_ms"`
	Status5xx      int       `json:"status_5xx"`
}

type State struct {
	mu      sync.RWMutex
	control Control
	stats   map[string]*EventStats
	counter uint64
	rngMu   sync.Mutex
	rng     *rand.Rand
}

func NewState() *State {
	return &State{
		control: Control{Mode: "none"},
		stats:   map[string]*EventStats{},
		rng:     rand.New(rand.NewSource(20261018)),
	}
}

func (s *State) setControl(c Control) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c.EventID == "" {
		c.EventID = "default"
	}
	s.control = c
	if _, ok := s.stats[c.EventID]; !ok {
		s.stats[c.EventID] = &EventStats{EventID: c.EventID}
	}
}

func (s *State) snapshotControl() Control {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.control
}

func (s *State) add(event string, duration float64, business, faulted, injectedErr bool, status int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	st, ok := s.stats[event]
	if !ok {
		st = &EventStats{EventID: event}
		s.stats[event] = st
	}
	st.Requests++
	if business {
		st.Business++
	}
	if faulted {
		st.Faulted++
	}
	if injectedErr {
		st.InjectedErrors++
	}
	if status >= 500 {
		st.Status5xx++
	}
	if business {
		st.DurationsMS = append(st.DurationsMS, duration)
		if len(st.DurationsMS) > 2048 {
			st.DurationsMS = st.DurationsMS[len(st.DurationsMS)-2048:]
		}
	}
}

func (s *State) getStats(event string) EventStats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if st, ok := s.stats[event]; ok {
		cp := *st
		cp.DurationsMS = append([]float64(nil), st.DurationsMS...)
		return cp
	}
	return EventStats{EventID: event}
}

func (s *State) random() float64 {
	s.rngMu.Lock()
	defer s.rngMu.Unlock()
	return s.rng.Float64()
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

func percentile(vals []float64, q float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := append([]float64(nil), vals...)
	sort.Float64s(cp)
	if len(cp) == 1 {
		return cp[0]
	}
	pos := q * float64(len(cp)-1)
	lo := int(pos)
	hi := lo + 1
	if hi >= len(cp) {
		return cp[lo]
	}
	f := pos - float64(lo)
	return cp[lo]*(1-f) + cp[hi]*f
}

func main() {
	upstreamRaw := os.Getenv("UPSTREAM")
	if upstreamRaw == "" {
		log.Fatal("UPSTREAM is required")
	}
	id := os.Getenv("PROXY_ID")
	if id == "" {
		id = "NF"
	}
	upstream, err := url.Parse(upstreamRaw)
	if err != nil {
		log.Fatal(err)
	}

	state := NewState()
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	origDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		origDirector(req)
		req.Host = upstream.Host
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		http.Error(w, "upstream unavailable", http.StatusBadGateway)
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		control := state.snapshotControl()
		event := control.EventID
		if event == "" {
			event = "default"
		}
		business := r.URL.Path != "/"
		var body []byte
		if r.Body != nil {
			body, _ = io.ReadAll(r.Body)
			r.Body.Close()
			r.Body = io.NopCloser(bytes.NewReader(body))
		}
		excluded := control.ExcludeToken != "" &&
			(strings.Contains(r.URL.Path, control.ExcludeToken) || bytes.Contains(body, []byte(control.ExcludeToken)))
		faulted := false
		injectedErr := false

		if business && !excluded && control.Mode != "" && control.Mode != "none" {
			n := atomic.AddUint64(&state.counter, 1)
			apply := true
			if control.ErrorEveryN > 0 {
				apply = int(n)%control.ErrorEveryN == 0
			}
			if control.FaultRate > 0 && control.FaultRate < 1 {
				apply = state.random() < control.FaultRate
			}
			if apply {
				faulted = true
				switch control.Mode {
				case "delay":
					if control.DelayMS > 0 {
						time.Sleep(time.Duration(control.DelayMS) * time.Millisecond)
					}
				case "jitter":
					upper := control.JitterMS
					if upper <= 0 {
						upper = control.DelayMS
					}
					if upper > 0 {
						d := int(state.random() * float64(upper))
						time.Sleep(time.Duration(d) * time.Millisecond)
					}
				case "error":
					injectedErr = true
					http.Error(w, "controlled business-path fault", http.StatusServiceUnavailable)
					state.add(event, float64(time.Since(start).Microseconds())/1000.0, business, faulted, injectedErr, http.StatusServiceUnavailable)
					return
				}
			}
		}

		sw := &statusWriter{ResponseWriter: w}
		proxy.ServeHTTP(sw, r)
		state.add(event, float64(time.Since(start).Microseconds())/1000.0, business, faulted, injectedErr, sw.status)
	})

	admin := http.NewServeMux()
	admin.HandleFunc("/control", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}
		var c Control
		if err := json.NewDecoder(r.Body).Decode(&c); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		state.setControl(c)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(c)
	})
	admin.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		event := r.URL.Query().Get("event_id")
		st := state.getStats(event)
		resp := map[string]any{
			"proxy_id":          id,
			"event_id":          st.EventID,
			"requests":          st.Requests,
			"business_requests": st.Business,
			"faulted_requests":  st.Faulted,
			"injected_errors":   st.InjectedErrors,
			"status_5xx":        st.Status5xx,
			"median_ms":         percentile(st.DurationsMS, 0.5),
			"p95_ms":            percentile(st.DurationsMS, 0.95),
			"max_ms":            percentile(st.DurationsMS, 1.0),
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})
	admin.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})

	go func() {
		s := &http.Server{Addr: ":9901", Handler: admin}
		log.Printf("%s admin on :9901", id)
		if err := s.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()

	server := &http.Server{
		Addr:              ":8000",
		Handler:           h2c.NewHandler(handler, &http2.Server{}),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("%s proxy listening on :8000 -> %s", id, upstream.String())
	log.Fatal(server.ListenAndServe())
}
