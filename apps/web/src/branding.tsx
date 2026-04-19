type IconProps = {
  className?: string;
};

export function BrandMark({ className }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="volta-mark-surface-dark" x1="7" y1="8" x2="57" y2="58" gradientUnits="userSpaceOnUse">
          <stop stopColor="#171513" />
          <stop offset="1" stopColor="#22201C" />
        </linearGradient>
        <linearGradient id="volta-mark-surface-light" x1="7" y1="8" x2="57" y2="58" gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFF9F1" />
          <stop offset="1" stopColor="#EFE5D8" />
        </linearGradient>
      </defs>
      <rect x="1.5" y="1.5" width="61" height="61" rx="17.5" fill="rgba(255,255,255,0.08)" />
      <rect x="6" y="6" width="52" height="52" rx="14" fill="url(#volta-mark-surface-dark)" />
      <path
        d="M18.8 40.3C24.1 40.3 27.6 36.8 27.6 30.9V18.8"
        stroke="#DDB16B"
        strokeWidth="4.8"
        strokeLinecap="round"
      />
      <path
        d="M34.4 43.2C41 43.2 44.6 38.4 44.6 31.4V18.8"
        stroke="#6280FF"
        strokeWidth="4.8"
        strokeLinecap="round"
      />
      <circle cx="30.2" cy="43.8" r="3.8" fill="#F5EEE4" />
    </svg>
  );
}

export function SunIcon({ className }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 2.75V5.1M12 18.9V21.25M21.25 12H18.9M5.1 12H2.75M18.55 5.45L16.9 7.1M7.1 16.9L5.45 18.55M18.55 18.55L16.9 16.9M7.1 7.1L5.45 5.45"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function MoonIcon({ className }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M18.15 14.75C16.9 15.35 15.5 15.68 14.02 15.68C8.76 15.68 4.5 11.42 4.5 6.16C4.5 5 4.71 3.88 5.08 2.85C2.74 4.2 1.15 6.74 1.15 9.65C1.15 13.96 4.64 17.45 8.95 17.45C12.14 17.45 14.9 15.53 16.11 12.78C16.57 13.55 17.26 14.24 18.15 14.75Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function InfoIcon({ className }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="8.25" stroke="currentColor" strokeWidth="1.8" />
      <path d="M12 10.3V16.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="12" cy="7.3" r="1.05" fill="currentColor" />
    </svg>
  );
}

export function CloseIcon({ className }: IconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M6.25 6.25L17.75 17.75M17.75 6.25L6.25 17.75"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}
