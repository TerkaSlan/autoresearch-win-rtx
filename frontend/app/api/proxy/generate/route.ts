import { NextRequest, NextResponse } from 'next/server';
import { checkAuth, getBackendHeaders, BACKEND_URL } from '../auth';

export async function POST(request: NextRequest) {
  // Apply rate limiting
  const authError = checkAuth(request);
  if (authError) return authError;

  try {
    const body = await request.json();

    // Remove checkpoint parameter - use globally loaded model instead
    const { checkpoint, ...generateBody } = body;

    const response = await fetch(`${BACKEND_URL}/generate`, {
      method: 'POST',
      headers: getBackendHeaders(),
      body: JSON.stringify(generateBody),
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend error: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to connect to backend' },
      { status: 502 }
    );
  }
}

// Reject other methods
export async function GET() {
  return NextResponse.json({ error: 'Method not allowed' }, { status: 405 });
}