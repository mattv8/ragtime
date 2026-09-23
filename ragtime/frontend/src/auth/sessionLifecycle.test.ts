import { describe, expect, it } from 'vitest';
import { createSessionLifecycle } from './sessionLifecycle';

describe('session lifecycle', () => {
  it('does not expire anonymous, public, or challenge requests', () => {
    const lifecycle = createSessionLifecycle();
    expect(lifecycle.expire(lifecycle.capture('session'))).toBeNull();
    expect(lifecycle.expire(lifecycle.capture('public'))).toBeNull();
    expect(lifecycle.expire(lifecycle.capture('challenge'))).toBeNull();
  });

  it('expires a current established session only once', () => {
    const lifecycle = createSessionLifecycle();
    const exchange = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(exchange, 'u1');
    const request = lifecycle.capture('session');
    expect(lifecycle.expire(request)).not.toBeNull();
    expect(lifecycle.expire(request)).toBeNull();
    expect(lifecycle.signedOutIntent).toBe(true);
  });

  it('rejects old requests after a terminal exchange and fences logout synchronously', () => {
    const lifecycle = createSessionLifecycle();
    const first = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(first, 'u1');
    const oldRequest = lifecycle.capture('session');
    const renewed = lifecycle.renewSession(lifecycle.capture('session'), 'u1')!;
    expect(lifecycle.expire(oldRequest)).toBeNull();
    const logout = lifecycle.beginLogout();
    expect(lifecycle.phase).toBe('logging-out');
    expect(lifecycle.adoptSession(renewed, 'u1')).toBeNull();
    expect(lifecycle.finishLogout(logout)).not.toBeNull();
  });

  it('adopts valid cookie bootstrap and retains same-user refresh generation', () => {
    const lifecycle = createSessionLifecycle();
    const bootstrap = lifecycle.capture('session');
    const adopted = lifecycle.adoptSession(bootstrap, 'u1')!;
    expect(adopted.generation).toBeGreaterThan(bootstrap.generation);
    const same = lifecycle.adoptSession(lifecycle.capture('session'), 'u1')!;
    expect(same.generation).toBe(adopted.generation);
  });

  it('fences authenticated work when a status result confirms anonymous', () => {
    const lifecycle = createSessionLifecycle();
    const established = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(established, 'u1');
    const privateRequest = lifecycle.capture('session');
    lifecycle.markAnonymous(lifecycle.capture('public'));
    expect(lifecycle.isCurrent(privateRequest)).toBe(false);
  });

  it('only clears signed-out intent after a settled logout receives verified anonymous status', () => {
    const lifecycle = createSessionLifecycle();
    const logout = lifecycle.beginLogout();
    lifecycle.finishLogout(logout);
    expect(lifecycle.signedOutIntent).toBe(true);
    lifecycle.markAnonymous(lifecycle.capture('public'));
    expect(lifecycle.signedOutIntent).toBe(false);

    const expired = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(expired, 'u1');
    lifecycle.expire(lifecycle.capture('session'));
    lifecycle.markAnonymous(lifecycle.capture('public'));
    expect(lifecycle.signedOutIntent).toBe(true);
  });

  it('keeps failed or pending logout fenced and isolates subscriber failures', () => {
    const lifecycle = createSessionLifecycle();
    const received: string[] = [];
    lifecycle.subscribe(() => {
      throw new Error('listener failure');
    });
    lifecycle.subscribe((event) => received.push(event.type));
    lifecycle.beginLogout();
    expect(received).toEqual(['logout-started']);
    expect(lifecycle.retrySignedOutSession()).toBeNull();
  });

  it('defers same-epoch expiry during renewal and applies it when renewal is abandoned', () => {
    const lifecycle = createSessionLifecycle();
    const establishing = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(establishing, 'u1');
    const oldRequest = lifecycle.capture('session');
    const renewal = lifecycle.beginRenewal(lifecycle.capture('session'))!;
    lifecycle.expire(oldRequest);
    expect(lifecycle.phase).toBe('authenticated');
    lifecycle.abandonRenewal(renewal);
    expect(lifecycle.phase).toBe('anonymous');
  });

  it('lets a completed renewal fence the concurrent old-cookie 401', () => {
    const lifecycle = createSessionLifecycle();
    const establishing = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(establishing, 'u1');
    const oldRequest = lifecycle.capture('session');
    const renewal = lifecycle.beginRenewal(lifecycle.capture('session'))!;
    lifecycle.expire(oldRequest);
    lifecycle.renewSession(renewal, 'u1');
    expect(lifecycle.phase).toBe('authenticated');
    expect(lifecycle.expire(oldRequest)).toBeNull();
  });

  it('queues reentrant events so observers receive the original transition first', () => {
    const lifecycle = createSessionLifecycle();
    const observed: string[] = [];
    lifecycle.subscribe((event) => {
      if (event.type === 'establishing') lifecycle.adoptSession(event.context, 'u1');
    });
    lifecycle.subscribe((event) => observed.push(event.type));

    lifecycle.beginEstablishment(lifecycle.capture('challenge'));

    expect(observed).toEqual(['establishing', 'adopted']);
  });

  it('fences a pre-adoption request when anonymous bootstrap discovers an account', () => {
    const lifecycle = createSessionLifecycle();
    lifecycle.markAnonymous(lifecycle.capture('public'));
    const beforeAdoption = lifecycle.capture('session');

    const adopted = lifecycle.adoptSession(lifecycle.capture('session'), 'u1')!;

    expect(adopted.generation).toBeGreaterThan(beforeAdoption.generation);
    expect(lifecycle.expire(beforeAdoption)).toBeNull();
  });

  it('does not allow passive adoption, stale revalidation, or stale logout completion to cross fences', () => {
    const lifecycle = createSessionLifecycle();
    const establishing = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(establishing, 'u1');
    const oldSession = lifecycle.capture('session');
    const logout = lifecycle.beginLogout();

    expect(lifecycle.adoptSession(oldSession, 'u2')).toBeNull();
    expect(lifecycle.revalidate(oldSession)).toBeNull();
    expect(lifecycle.finishLogout(oldSession)).toBeNull();
    expect(lifecycle.finishLogout(logout)).not.toBeNull();
    expect(lifecycle.retrySignedOutSession()).not.toBeNull();
  });

  it('defers anonymous status during renewal, while explicit logout still wins', () => {
    const lifecycle = createSessionLifecycle();
    const establishing = lifecycle.beginEstablishment(lifecycle.capture('challenge'))!;
    lifecycle.adoptSession(establishing, 'u1');
    const renewal = lifecycle.beginRenewal(lifecycle.capture('session'))!;

    lifecycle.markAnonymous(lifecycle.capture('public'));
    expect(lifecycle.phase).toBe('authenticated');
    lifecycle.beginLogout();

    expect(lifecycle.phase).toBe('logging-out');
    expect(lifecycle.renewSession(renewal, 'u1')).toBeNull();
  });
});
