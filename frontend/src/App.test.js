import { render, screen } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ status: 'ok' })
    })
  );
});

afterEach(() => {
  jest.resetAllMocks();
});

test('renders the redesigned detector hero', async () => {
  render(<App />);
  expect(screen.getByText(/analyze media for manipulation signals/i)).toBeInTheDocument();
  expect(screen.getByText(/start with a file/i)).toBeInTheDocument();
  expect((await screen.findAllByText(/api online/i)).length).toBeGreaterThan(0);
});
