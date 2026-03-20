import { NextRequest, NextResponse } from 'next/server';
import { checkAuth, getBackendHeaders, BACKEND_URL } from '../../auth';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ exp_dir: string }> }
) {
  // Apply rate limiting
  const authError = checkAuth(request);
  if (authError) return authError;

  const { exp_dir } = await params;
  try {
    const response = await fetch(`${BACKEND_URL}/info/${exp_dir}`, {
      cache: 'no-store',
      headers: getBackendHeaders(),
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