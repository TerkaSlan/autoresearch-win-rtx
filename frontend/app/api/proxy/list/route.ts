import { NextRequest, NextResponse } from 'next/server';
import { checkAuth, getBackendHeaders, BACKEND_URL } from '../auth';

export async function GET(request: NextRequest) {
  // Apply rate limiting
  const authError = checkAuth(request);
  if (authError) return authError;

  try {
    // Our API serves pre-computed samples from experiment directories
    const response = await fetch(`${BACKEND_URL}/experiments`, {
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

    // Transform API response to match frontend expectations
    const transformedData = {
      experiments: data.map((exp: any) => ({
        exp_dir: exp.name,
        checkpoint_path: exp.checkpoint_exists ? `${BACKEND_URL}/checkpoints/${exp.name}/checkpoint.pt` : null,
        has_checkpoint: exp.checkpoint_exists,
        has_program: true,
        has_results: true,
        has_git_log: exp.git_log !== null,
        has_global_results: true,
        has_metadata: exp.model_info !== null,
        git_log: exp.git_log ? [{
          commit_sha: '',
          commit_short: exp.name.split('-')[1] || '',
          author: '',
          date: '',
          message: ''
        }] : [],
        results_data: null,
        metadata: exp.model_info ? parseModelInfo(exp.model_info) : null,
        sample_output: exp.sample_output
      })),
      count: data.length
    };

    return NextResponse.json(transformedData);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to connect to backend' },
      { status: 502 }
    );
  }
}

// Parse model_info.txt content into a structured object
function parseModelInfo(text: string): Record<string, any> {
  const result: Record<string, any> = {};
  text.split('\n').forEach(line => {
    const [key, ...valueParts] = line.split(':');
    if (key && valueParts.length > 0) {
      result[key.trim()] = valueParts.join(':').trim();
    }
  });
  return result;
}