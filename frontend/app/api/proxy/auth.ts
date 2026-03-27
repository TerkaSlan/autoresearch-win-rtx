import { NextRequest, NextResponse } from 'next/server';

// Backend URL from environment (for K8s cluster internal communication)
export const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// API key for backend authentication - SERVER SIDE ONLY
const API_KEY = process.env.AUTORESEARCH_API_KEY;

// Rate limiting - simple in-memory store (resets on server restart)
// For production, use Redis or similar
const rateLimitStore = new Map<string, { count: number; resetTime: number }>();
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = 500; // 500 requests per minute per IP (~45 page loads)

interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetTime: number;
}

function checkRateLimit(ip: string): RateLimitResult {
  const now = Date.now();
  const entry = rateLimitStore.get(ip);

  if (!entry || now > entry.resetTime) {
    // New window
    rateLimitStore.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW_MS });
    return { allowed: true, remaining: RATE_LIMIT_MAX_REQUESTS - 1, resetTime: now + RATE_LIMIT_WINDOW_MS };
  }

  if (entry.count >= RATE_LIMIT_MAX_REQUESTS) {
    return { allowed: false, remaining: 0, resetTime: entry.resetTime };
  }

  entry.count++;
  return { allowed: true, remaining: RATE_LIMIT_MAX_REQUESTS - entry.count, resetTime: entry.resetTime };
}

/**
 * Apply rate limiting to incoming requests.
 * Returns null if allowed, or an error response if rate limited.
 */
export function applyRateLimit(request: NextRequest): NextResponse | null {
  // Get client IP - check various headers for proxied requests
  const forwardedFor = request.headers.get('x-forwarded-for');
  const realIp = request.headers.get('x-real-ip');
  const ip = forwardedFor?.split(',')[0]?.trim() || realIp || 'unknown';

  const result = checkRateLimit(ip);

  if (!result.allowed) {
    return NextResponse.json(
      { error: 'Too Many Requests', message: 'Rate limit exceeded', retryAfter: Math.ceil((result.resetTime - Date.now()) / 1000) },
      { status: 429, headers: { 'Retry-After': String(Math.ceil((result.resetTime - Date.now()) / 1000)) } }
    );
  }

  return null;
}

/**
 * Get headers for backend API calls (includes API key).
 * This is used when the Next.js server calls the backend.
 */
export function getBackendHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (API_KEY) {
    headers['Authorization'] = `Bearer ${API_KEY}`;
  }

  return headers;
}

/**
 * Check auth for incoming requests - only rate limiting.
 * API key validation is done server-side when calling backend.
 */
export function checkAuth(request: NextRequest): NextResponse | null {
  // Only apply rate limiting for incoming requests
  return applyRateLimit(request);
}